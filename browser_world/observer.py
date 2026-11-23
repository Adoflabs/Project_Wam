"""Four-plane Chromium observer: structured page state, no screenshots.

The observer turns a live Playwright page into an `Observation` with four
planes, read through the Chrome DevTools Protocol (CDP):

  * Semantic: accessibility role, name, value and states (invalid, required,
    disabled, checked, ...) for each relevant element, plus its parent.
  * Geometry: each element's box, whether it is in the viewport, and whether
    a click at its centre would actually reach it (hit testing).
  * Motion: viewport, scroll position, page size, and which elements are
    pinned (fixed or sticky) and in what order they are painted.
  * Temporal: DOM mutations, navigations and network activity since the
    previous observation, and whether the page has settled.

No pixels and no HTML are captured. Element states come from Chromium's
accessibility tree, geometry from one DOM snapshot, and hit tests from
`DOM.getNodeForLocation`.

Leakage rule: the temporal plane counts what happened *since the previous
observation*. In a transition (before, action, after) those counts belong to
`after` and must never be used as input when predicting that action's effect.

Typical usage example:

  watcher = observer.attach(page)  # Before page.goto().
  page.goto(url)
  watcher.wait_until_settled()
  observation = watcher.observe()
  print(observation.to_text())
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import re
import time
from typing import Any

from playwright import sync_api

SEMANTIC = 'semantic'
GEOMETRY = 'geometry'
MOTION = 'motion'
TEMPORAL = 'temporal'
ALL_PLANES = (SEMANTIC, GEOMETRY, MOTION, TEMPORAL)

# Roles a person (or an agent) acts on.
INTERACTIVE_ROLES = frozenset({
    'button', 'checkbox', 'combobox', 'link', 'listbox', 'menuitem',
    'option', 'radio', 'searchbox', 'slider', 'spinbutton', 'switch', 'tab',
    'textbox',
})
# Roles that give structure or report changes.
STRUCTURAL_ROLES = frozenset({
    'alert', 'alertdialog', 'banner', 'dialog', 'form', 'group', 'heading',
    'navigation', 'radiogroup', 'status', 'tabpanel',
})
# Roles that only carry page text; kept when `include_text` is set.
TEXT_ROLES = frozenset({'listitem', 'paragraph'})
# Roles whose accessible name Chromium leaves empty, so the observer reads
# their text content instead.
_NAME_FROM_CONTENT_ROLES = frozenset(
    {'alert', 'banner', 'listitem', 'paragraph', 'status'})

STATE_NAMES = ('checked', 'disabled', 'expanded', 'focused', 'invalid',
               'readonly', 'required', 'selected')

_MATCH_KEY_ATTRIBUTE = 'data-wam-key'
_MAX_TEXT_LENGTH = 120
_WHITESPACE_RE = re.compile(r'\s+')
_PINNED_POSITIONS = frozenset({'fixed', 'sticky'})
# Order matters: _DomSnapshot reads the styles by position.
_SNAPSHOT_STYLES = ('position', 'visibility', 'opacity')
_HIDDEN_VISIBILITY = frozenset({'hidden', 'collapse'})
_OBJECT_GROUP = 'wam-observer'

# Counts DOM mutations in the page. It touches only a private window
# property, never the DOM, so it cannot change what the page shows.
# `mutations` is reset by each observation; `total` only ever grows and is
# the document's version number.
_TEMPORAL_INIT_SCRIPT = """
(() => {
  const state = {mutations: 0, total: 0, last: performance.now()};
  new MutationObserver((records) => {
    state.mutations += records.length;
    state.total += records.length;
    state.last = performance.now();
  }).observe(document, {subtree: true, childList: true, attributes: true,
                        characterData: true});
  Object.defineProperty(window, '__wamTemporal', {value: {
    take() {
      const out = {mutations: state.mutations, total: state.total,
                   msSinceMutation: performance.now() - state.last};
      state.mutations = 0;
      return out;
    },
    peek() {
      return {mutations: state.mutations, total: state.total,
              msSinceMutation: performance.now() - state.last};
    },
  }});
})();
"""
_NO_TEMPORAL = '{mutations: 0, total: 0, msSinceMutation: 0}'


@dataclasses.dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in CSS pixels, in document coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        """The rectangle's centre point."""
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclasses.dataclass(frozen=True)
class Element:
    """One observed element, with fields from the semantic, geometry and
    motion planes.

    Attributes:
      node_id: Chromium's backend DOM node id. Stable while the node lives;
        a re-rendered element gets a new id.
      role: Accessibility role, such as 'textbox'.
      name: Accessible name, or text content for text-like roles.
      value: Current value for fields, or None.
      states: Accessibility states, keyed by `STATE_NAMES`.
      parent_id: `node_id` of the nearest observed ancestor, or None.
      match_key: The element's `data-wam-key`. Used only to match elements
        across observations when building training labels; never model
        input, because real websites do not have it.
      box: Layout box, or None for elements without layout (such as the
        options of a closed dropdown).
      visible: Whether the element is rendered with a non-empty box and is
        not hidden by `visibility` or zero opacity (its own or an
        ancestor's). Says nothing about being covered or scrolled away.
      in_viewport: Whether the box intersects the visible viewport.
      hit_point: For interactive elements in the viewport, the document
        point that was hit-tested: the centre of the box's visible part.
        Actions should press here. Otherwise None.
      hit_testable: Whether a click at `hit_point` reaches this element, or
        None when no hit test ran.
      obstructed: Whether the hit test ran and reached a different element
        (an overlay, or a scroll container clipping this element).
      clickable: Whether Chromium reports a click listener or native click
        behaviour for the element.
      pinned: Whether the element or an ancestor is fixed or sticky, so it
        does not move when the page scrolls.
      paint_order: Chromium's paint order; higher is drawn on top.
    """

    node_id: int
    role: str
    name: str
    value: str | None
    states: Mapping[str, bool]
    parent_id: int | None
    match_key: str | None
    box: Box | None
    visible: bool
    in_viewport: bool
    hit_point: tuple[float, float] | None
    hit_testable: bool | None
    obstructed: bool
    clickable: bool
    pinned: bool
    paint_order: int | None

    def to_dict(self, planes: Iterable[str] = ALL_PLANES) -> dict[str, Any]:
        """Returns a JSON-ready dict with the fields of the given planes.

        `node_id` and `match_key` are always included so records can be
        matched; the training pipeline must drop `match_key` from inputs.
        """
        planes = frozenset(planes)
        out: dict[str, Any] = {'node_id': self.node_id,
                               'match_key': self.match_key}
        if SEMANTIC in planes:
            out.update(role=self.role, name=self.name, value=self.value,
                       states=dict(self.states), parent_id=self.parent_id,
                       clickable=self.clickable)
        if GEOMETRY in planes:
            out.update(
                box=dataclasses.asdict(self.box) if self.box else None,
                visible=self.visible,
                in_viewport=self.in_viewport,
                hit_point=(list(self.hit_point) if self.hit_point else None),
                hit_testable=self.hit_testable,
                obstructed=self.obstructed)
        if MOTION in planes:
            out.update(pinned=self.pinned, paint_order=self.paint_order)
        return out


@dataclasses.dataclass(frozen=True)
class Motion:
    """Page-level motion plane.

    Attributes:
      viewport_width: Visible width in CSS pixels.
      viewport_height: Visible height in CSS pixels.
      scroll_x: Horizontal scroll offset.
      scroll_y: Vertical scroll offset.
      content_width: Full document width.
      content_height: Full document height.
      pinned_count: Number of observed elements that are fixed or sticky.
      stacking_contexts: Number of stacking contexts (overlay layers).
    """

    viewport_width: float
    viewport_height: float
    scroll_x: float
    scroll_y: float
    content_width: float
    content_height: float
    pinned_count: int
    stacking_contexts: int


@dataclasses.dataclass(frozen=True)
class Temporal:
    """Page-level temporal plane, covering the time since the previous
    observation.

    Attributes:
      timestamp_ms: Wall-clock time of this observation, in milliseconds.
      mutations: DOM mutation records since the previous observation.
      ms_since_mutation: Time since the latest DOM mutation.
      navigations: Main-frame navigations since the previous observation.
      requests_started: Network requests started since then.
      pending_requests: Requests still in flight now.
      settled: Whether `wait_until_settled` succeeded before this
        observation; None if it was not called.
    """

    timestamp_ms: float
    mutations: int
    ms_since_mutation: float
    navigations: int
    requests_started: int
    pending_requests: int
    settled: bool | None


@dataclasses.dataclass(frozen=True)
class Observation:
    """Everything the observer saw at one moment.

    Attributes:
      sequence: This observer's observation counter, starting at 1.
      page_epoch: Main-frame navigations seen before this observation.
      dom_version: DOM mutations in the current document so far. Together
        with `page_epoch` it identifies the page version observed.
      url: The page URL.
      title: The document title.
      elements: Observed elements in document order.
      motion: Page-level motion plane.
      temporal: Page-level temporal plane.
    """

    sequence: int
    page_epoch: int
    dom_version: int
    url: str
    title: str
    elements: tuple[Element, ...]
    motion: Motion
    temporal: Temporal

    def find(self, match_key: str) -> Element | None:
        """Returns the element with `match_key`, or None. For tests and
        label building only."""
        for element in self.elements:
            if element.match_key == match_key:
                return element
        return None

    def to_dict(self, planes: Iterable[str] = ALL_PLANES) -> dict[str, Any]:
        """Returns a JSON-ready dict containing only the given planes."""
        planes = frozenset(planes)
        out: dict[str, Any] = {
            'sequence': self.sequence,
            'page_epoch': self.page_epoch,
            'dom_version': self.dom_version,
            'url': self.url,
            'title': self.title,
            'elements': [e.to_dict(planes) for e in self.elements],
        }
        if MOTION in planes:
            out['motion'] = dataclasses.asdict(self.motion)
        if TEMPORAL in planes:
            out['temporal'] = dataclasses.asdict(self.temporal)
        return out

    def to_text(self, planes: Iterable[str] = ALL_PLANES) -> str:
        """Returns a compact, human-readable rendering for debugging."""
        planes = frozenset(planes)
        lines = [f'URL {self.url}  TITLE "{self.title}"']
        for element in self.elements:
            lines.append(_element_line(element, planes))
        if MOTION in planes:
            m = self.motion
            lines.append(
                f'MOTION viewport={m.viewport_width:.0f}x'
                f'{m.viewport_height:.0f} scroll={m.scroll_x:.0f},'
                f'{m.scroll_y:.0f} pinned={m.pinned_count} '
                f'layers={m.stacking_contexts}')
        if TEMPORAL in planes:
            t = self.temporal
            lines.append(
                f'TEMPORAL mutations={t.mutations} '
                f'quiet_ms={t.ms_since_mutation:.0f} '
                f'navigations={t.navigations} '
                f'requests={t.requests_started} pending={t.pending_requests}'
                f' settled={t.settled}')
        return '\n'.join(lines)


def _element_line(element: Element, planes: frozenset[str]) -> str:
    parts = [f'[{element.node_id}]']
    if SEMANTIC in planes:
        parts.append(f'{element.role} "{element.name}"')
        if element.value is not None:
            parts.append(f'value="{element.value}"')
        flags = [name for name in STATE_NAMES if element.states.get(name)]
        if flags:
            parts.append('{' + ','.join(flags) + '}')
    if GEOMETRY in planes and element.box is not None:
        b = element.box
        parts.append(f'@{b.x:.0f},{b.y:.0f} {b.width:.0f}x{b.height:.0f}')
        if not element.visible:
            parts.append('invisible')
        if not element.in_viewport:
            parts.append('offscreen')
        if element.obstructed:
            parts.append('obstructed')
    if MOTION in planes and element.pinned:
        parts.append('pinned')
    return ' '.join(parts)


def _ax_value(node: Mapping[str, Any], field: str) -> Any:
    return node.get(field, {}).get('value')


def _clean_text(text: str) -> str:
    text = _WHITESPACE_RE.sub(' ', text).strip()
    if len(text) > _MAX_TEXT_LENGTH:
        return text[:_MAX_TEXT_LENGTH - 1] + '…'
    return text


def _as_bool(value: Any) -> bool:
    """Normalizes CDP state values ('true', 'false', 'mixed', bools)."""
    if isinstance(value, str):
        return value not in ('false', '')
    return bool(value)


class _DomSnapshot:
    """Index over one `DOMSnapshot.captureSnapshot` result for the main
    document, keyed by backend node id."""

    def __init__(self, snapshot: Mapping[str, Any]):
        strings = snapshot['strings']
        document = snapshot['documents'][0]
        nodes = document['nodes']
        layout = document['layout']
        self.scroll_x = float(document.get('scrollOffsetX', 0))
        self.scroll_y = float(document.get('scrollOffsetY', 0))
        self.content_width = float(document.get('contentWidth', 0))
        self.content_height = float(document.get('contentHeight', 0))
        self.stacking_contexts = len(
            layout.get('stackingContexts', {}).get('index', []))

        self._parent = nodes['parentIndex']
        self._index_of = {
            backend_id: index
            for index, backend_id in enumerate(nodes['backendNodeId'])
        }
        self._backend_of = nodes['backendNodeId']
        self._clickable = frozenset(nodes.get('isClickable', {}).get(
            'index', []))
        self._match_key: dict[int, str] = {}
        for index, attributes in enumerate(nodes['attributes']):
            for i in range(0, len(attributes), 2):
                if strings[attributes[i]] == _MATCH_KEY_ATTRIBUTE:
                    self._match_key[index] = strings[attributes[i + 1]]
        self._required = frozenset(
            index for index, attributes in enumerate(nodes['attributes'])
            if _declares_required(attributes, strings))

        self._box: dict[int, Box] = {}
        self._paint_order: dict[int, int] = {}
        self._own_pinned: set[int] = set()
        self._hidden: set[int] = set()
        self._transparent: set[int] = set()
        paint_orders = layout.get('paintOrders', [])
        for i, node_index in enumerate(layout['nodeIndex']):
            x, y, width, height = layout['bounds'][i]
            self._box[node_index] = Box(x, y, width, height)
            if i < len(paint_orders):
                self._paint_order[node_index] = paint_orders[i]
            styles = layout['styles'][i]
            if not styles:
                continue  # Text runs carry no computed styles of their own.
            position, visibility, opacity = (strings[s] for s in styles)
            if position in _PINNED_POSITIONS:
                self._own_pinned.add(node_index)
            if visibility in _HIDDEN_VISIBILITY:
                self._hidden.add(node_index)
            if opacity == '0':
                self._transparent.add(node_index)

    def index(self, backend_id: int) -> int | None:
        """Returns the snapshot index of a backend node, or None."""
        return self._index_of.get(backend_id)

    def box(self, index: int) -> Box | None:
        """Returns the node's layout box, or None without layout."""
        return self._box.get(index)

    def paint_order(self, index: int) -> int | None:
        """Returns the node's paint order, or None without layout."""
        return self._paint_order.get(index)

    def match_key(self, index: int) -> str | None:
        """Returns the node's `data-wam-key` attribute, or None."""
        return self._match_key.get(index)

    def clickable(self, index: int) -> bool:
        """Returns whether Chromium marks the node clickable."""
        return index in self._clickable

    def required(self, index: int) -> bool:
        """Returns whether the node's attributes mark it required."""
        return index in self._required

    def visible(self, index: int) -> bool:
        """Returns whether the node is rendered and not see-through.

        `visibility` is inherited, so the node's own value suffices;
        opacity is not, so a fully transparent ancestor also hides it.
        """
        box = self._box.get(index)
        if box is None or box.width <= 0 or box.height <= 0:
            return False
        if index in self._hidden:
            return False
        while index != -1:
            if index in self._transparent:
                return False
            index = self._parent[index]
        return True

    def pinned(self, index: int) -> bool:
        """Returns whether the node or an ancestor is fixed or sticky."""
        while index != -1:
            if index in self._own_pinned:
                return True
            index = self._parent[index]
        return False

    def is_self_or_descendant(self, backend_id: int, ancestor: int) -> bool:
        """Returns whether node `backend_id` is node `ancestor` or inside
        it; both are backend node ids."""
        index = self._index_of.get(backend_id)
        while index is not None and index != -1:
            if self._backend_of[index] == ancestor:
                return True
            index = self._parent[index]
        return False


class Observer:
    """Reads four-plane observations from one Playwright page.

    Create it with `attach()` before the page navigates, so the temporal
    script is installed in every document the page loads.
    """

    def __init__(self, page: sync_api.Page, include_text: bool,
                 cull_offscreen: bool):
        """Connects to the page over CDP. Use `attach()` instead.

        Args:
          page: The page to observe.
          include_text: Whether to keep text-only roles (paragraphs and list
            items).
          cull_offscreen: Whether to drop elements entirely outside the
            viewport. Their `in_viewport` flag is kept either way.
        """
        self._page = page
        self._include_text = include_text
        self._cull_offscreen = cull_offscreen
        self._pending_requests: set[str] = set()
        self._requests_started = 0
        self._navigations = 0
        self._page_epoch = 0
        self._sequence = 0
        self._settled: bool | None = None
        self._cdp = page.context.new_cdp_session(page)
        self._cdp.on('Network.requestWillBeSent', self._on_request)
        self._cdp.on('Network.loadingFinished', self._on_request_done)
        self._cdp.on('Network.loadingFailed', self._on_request_done)
        self._cdp.on('Page.frameNavigated', self._on_navigated)
        for domain in ('Accessibility', 'DOM', 'DOMSnapshot', 'Network',
                       'Page'):
            self._cdp.send(f'{domain}.enable')
        page.add_init_script(_TEMPORAL_INIT_SCRIPT)

    def _on_request(self, event: Mapping[str, Any]) -> None:
        self._pending_requests.add(event['requestId'])
        self._requests_started += 1

    def _on_request_done(self, event: Mapping[str, Any]) -> None:
        self._pending_requests.discard(event['requestId'])

    def _on_navigated(self, event: Mapping[str, Any]) -> None:
        if not event['frame'].get('parentId'):
            self._navigations += 1
            self._page_epoch += 1
            self._pending_requests.clear()

    def wait_until_settled(self, quiet_ms: float = 100,
                           timeout_ms: float = 5000) -> bool:
        """Waits until no requests are in flight and the DOM is quiet.

        Args:
          quiet_ms: Required time without DOM mutations.
          timeout_ms: Maximum time to wait.

        Returns:
          True if the page settled before the timeout. The result is also
          recorded in the next observation's temporal plane.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            # Waiting inside Playwright lets it deliver queued CDP events.
            self._page.wait_for_timeout(20)
            quiet = self._page.evaluate(
                'window.__wamTemporal ? '
                'window.__wamTemporal.peek().msSinceMutation : 0')
            if not self._pending_requests and quiet >= quiet_ms:
                self._settled = True
                return True
            if time.monotonic() >= deadline:
                self._settled = False
                return False

    def observe(self) -> Observation:
        """Captures all four planes and resets the temporal counters."""
        ax_nodes = self._cdp.send('Accessibility.getFullAXTree')['nodes']
        snapshot = _DomSnapshot(self._cdp.send(
            'DOMSnapshot.captureSnapshot', {
                'computedStyles': list(_SNAPSHOT_STYLES),
                'includePaintOrder': True,
                'includeDOMRects': True,
            }))
        metrics = self._cdp.send('Page.getLayoutMetrics')['cssVisualViewport']
        viewport = Box(float(metrics['pageX']), float(metrics['pageY']),
                       float(metrics['clientWidth']),
                       float(metrics['clientHeight']))
        elements = self._elements(ax_nodes, snapshot, viewport)
        if self._cull_offscreen:
            elements = [e for e in elements if e.in_viewport or e.box is None]
        temporal, dom_version = self._take_temporal()
        self._sequence += 1
        motion = Motion(
            viewport_width=viewport.width,
            viewport_height=viewport.height,
            scroll_x=viewport.x,
            scroll_y=viewport.y,
            content_width=snapshot.content_width,
            content_height=snapshot.content_height,
            pinned_count=sum(e.pinned for e in elements),
            stacking_contexts=snapshot.stacking_contexts,
        )
        return Observation(
            sequence=self._sequence,
            page_epoch=self._page_epoch,
            dom_version=dom_version,
            url=self._page.url,
            title=self._page.title(),
            elements=tuple(elements),
            motion=motion,
            temporal=temporal,
        )

    def current_version(self) -> tuple[int, int]:
        """Returns the live (page_epoch, dom_version) of the page.

        Compare with an observation's fields to tell whether the DOM or the
        document changed since it was taken. Typing into a field changes
        its value without a DOM mutation, so this cannot detect that.
        """
        # Evaluating first lets Playwright deliver pending navigation events.
        counts = self._page.evaluate(
            f'window.__wamTemporal ? window.__wamTemporal.peek() : '
            f'{_NO_TEMPORAL}')
        return self._page_epoch, int(counts['total'])

    def is_current(self, observation: Observation) -> bool:
        """Returns whether the page still matches `observation`: same
        document, no DOM mutation since, and the same scroll position."""
        version = (observation.page_epoch, observation.dom_version)
        if self.current_version() != version:
            return False
        metrics = self._cdp.send('Page.getLayoutMetrics')['cssVisualViewport']
        return (float(metrics['pageX']) == observation.motion.scroll_x
                and float(metrics['pageY']) == observation.motion.scroll_y)

    def hits_node(self, x: float, y: float, node_id: int) -> bool | None:
        """Returns whether a click at document point (x, y) reaches a node.

        A hit on any descendant counts, as a click on a button's text
        activates the button.

        Args:
          x: Document x coordinate, inside the visible viewport.
          y: Document y coordinate, inside the visible viewport.
          node_id: Backend node id of the intended target.

        Returns:
          True or False, or None if the target node no longer exists.
        """
        try:
            target = self._cdp.send('DOM.resolveNode', {
                'backendNodeId': node_id, 'objectGroup': _OBJECT_GROUP,
            })['object']['objectId']
        except sync_api.Error:
            return None
        try:
            hit_id = self._cdp.send('DOM.getNodeForLocation', {
                'x': round(x), 'y': round(y),
                'includeUserAgentShadowDOM': False,
            })['backendNodeId']
            if hit_id == node_id:
                return True
            hit = self._cdp.send('DOM.resolveNode', {
                'backendNodeId': hit_id, 'objectGroup': _OBJECT_GROUP,
            })['object']['objectId']
            return bool(self._cdp.send('Runtime.callFunctionOn', {
                'objectId': target,
                'functionDeclaration':
                    'function(n) { return this.contains(n); }',
                'arguments': [{'objectId': hit}],
                'returnByValue': True,
            })['result']['value'])
        except sync_api.Error:
            return False  # Nothing hittable at that point.
        finally:
            self._cdp.send('Runtime.releaseObjectGroup',
                           {'objectGroup': _OBJECT_GROUP})

    def is_enabled(self, node_id: int) -> bool | None:
        """Returns whether a node is enabled right now, from the live
        accessibility tree, or None if the node no longer exists."""
        try:
            nodes = self._cdp.send('Accessibility.getPartialAXTree', {
                'backendNodeId': node_id, 'fetchRelatives': False,
            })['nodes']
        except sync_api.Error:
            return None
        for prop in nodes[0].get('properties', []) if nodes else []:
            if prop['name'] == 'disabled':
                return not _as_bool(prop.get('value', {}).get('value'))
        return True

    def _take_temporal(self) -> tuple[Temporal, int]:
        """Returns the temporal plane and the DOM version, and resets the
        per-observation counters."""
        page_counts = self._page.evaluate(
            f'window.__wamTemporal ? window.__wamTemporal.take() : '
            f'{_NO_TEMPORAL}')
        temporal = Temporal(
            timestamp_ms=time.time() * 1000,
            mutations=int(page_counts['mutations']),
            ms_since_mutation=float(page_counts['msSinceMutation']),
            navigations=self._navigations,
            requests_started=self._requests_started,
            pending_requests=len(self._pending_requests),
            settled=self._settled,
        )
        self._navigations = 0
        self._requests_started = 0
        self._settled = None
        return temporal, int(page_counts['total'])

    def _kept_roles(self) -> frozenset[str]:
        roles = INTERACTIVE_ROLES | STRUCTURAL_ROLES
        return roles | TEXT_ROLES if self._include_text else roles

    def _elements(self, ax_nodes: Sequence[Mapping[str, Any]],
                  snapshot: _DomSnapshot, viewport: Box) -> list[Element]:
        """Walks the accessibility tree in document order and builds the
        kept elements."""
        by_id = {node['nodeId']: node for node in ax_nodes}
        kept_roles = self._kept_roles()
        elements: list[Element] = []
        # Depth-first walk; the stack holds (AX node id, nearest kept
        # ancestor's backend id). CDP lists nodes out of document order, so
        # the tree is walked from the root through childIds.
        stack: list[tuple[str, int | None]] = [(ax_nodes[0]['nodeId'], None)]
        while stack:
            node_id, parent_backend = stack.pop()
            node = by_id.get(node_id)
            if node is None:
                continue
            child_parent = parent_backend
            role = _ax_value(node, 'role')
            backend_id = node.get('backendDOMNodeId')
            if (not node.get('ignored') and role in kept_roles
                    and backend_id is not None):
                element = self._element(node, by_id, snapshot, viewport,
                                        parent_backend)
                if element is not None:
                    elements.append(element)
                    child_parent = backend_id
            for child in reversed(node.get('childIds', [])):
                stack.append((child, child_parent))
        return elements

    def _element(self, node: Mapping[str, Any],
                 by_id: Mapping[str, Mapping[str, Any]],
                 snapshot: _DomSnapshot, viewport: Box,
                 parent_backend: int | None) -> Element | None:
        role = _ax_value(node, 'role')
        backend_id = node['backendDOMNodeId']
        name = _clean_text(_ax_value(node, 'name') or '')
        if not name and role in _NAME_FROM_CONTENT_ROLES:
            name = _clean_text(_descendant_text(node, by_id))
            if not name:
                return None  # An empty live region carries no information.
        properties = {
            prop['name']: prop.get('value', {}).get('value')
            for prop in node.get('properties', [])
        }
        index = snapshot.index(backend_id)
        states = {
            state: _as_bool(properties.get(state, False))
            for state in STATE_NAMES
        }
        if index is not None and snapshot.required(index):
            states['required'] = True
        box = snapshot.box(index) if index is not None else None
        in_viewport = box is not None and _intersects(box, viewport)
        visible = index is not None and snapshot.visible(index)
        hit_point = None
        hit_testable = None
        if box is not None and in_viewport and role in INTERACTIVE_ROLES:
            hit_point = _visible_center(box, viewport)
            hit_testable = self._hit_test(hit_point, backend_id, snapshot)
        value = _ax_value(node, 'value')
        return Element(
            node_id=backend_id,
            role=role,
            name=name,
            value=None if value is None else _clean_text(str(value)),
            states=states,
            parent_id=parent_backend,
            match_key=snapshot.match_key(index) if index is not None else None,
            box=box,
            visible=visible,
            in_viewport=in_viewport,
            hit_point=hit_point,
            hit_testable=hit_testable,
            obstructed=hit_testable is False,
            clickable=index is not None and snapshot.clickable(index),
            pinned=index is not None and snapshot.pinned(index),
            paint_order=(snapshot.paint_order(index)
                         if index is not None else None),
        )

    def _hit_test(self, point: tuple[float, float], backend_id: int,
                  snapshot: _DomSnapshot) -> bool:
        """Returns whether a click at document `point` reaches the element
        or one of its descendants."""
        # getNodeForLocation takes document coordinates, but only finds
        # nodes inside the visible viewport.
        hit = self._cdp.send('DOM.getNodeForLocation', {
            'x': round(point[0]),
            'y': round(point[1]),
            'includeUserAgentShadowDOM': False,
        })
        return snapshot.is_self_or_descendant(hit['backendNodeId'],
                                              backend_id)


def _descendant_text(node: Mapping[str, Any],
                     by_id: Mapping[str, Mapping[str, Any]]) -> str:
    """Concatenates the names of all StaticText descendants of `node`."""
    parts = []
    stack = list(reversed(node.get('childIds', [])))
    while stack:
        child = by_id.get(stack.pop())
        if child is None or child.get('ignored'):
            continue
        if _ax_value(child, 'role') == 'StaticText':
            parts.append(_ax_value(child, 'name') or '')
            continue  # Its InlineTextBox children repeat the same text.
        stack.extend(reversed(child.get('childIds', [])))
    return ' '.join(parts)


def _declares_required(attributes: Sequence[int],
                       strings: Sequence[str]) -> bool:
    """Returns whether a snapshot attribute list marks a field required."""
    for i in range(0, len(attributes), 2):
        name = strings[attributes[i]]
        value = strings[attributes[i + 1]]
        if name == 'required' or (name == 'aria-required' and value == 'true'):
            return True
    return False


def _visible_center(box: Box, viewport: Box) -> tuple[float, float]:
    """Returns the centre of the part of `box` inside `viewport`, in
    document coordinates. Assumes they intersect."""
    left = max(box.x, viewport.x)
    right = min(box.x + box.width, viewport.x + viewport.width)
    top = max(box.y, viewport.y)
    bottom = min(box.y + box.height, viewport.y + viewport.height)
    return ((left + right) / 2, (top + bottom) / 2)


def _intersects(box: Box, viewport: Box) -> bool:
    return (box.width > 0 and box.height > 0
            and box.x < viewport.x + viewport.width
            and box.x + box.width > viewport.x
            and box.y < viewport.y + viewport.height
            and box.y + box.height > viewport.y)


def attach(page: sync_api.Page, include_text: bool = True,
           cull_offscreen: bool = False) -> Observer:
    """Starts observing a page. Call before the page navigates.

    Args:
      page: The page to observe.
      include_text: Whether to keep text-only elements (paragraphs and list
        items). On by default because the booking world reports outcomes,
        such as the confirmation reference, in plain text.
      cull_offscreen: Whether to drop elements entirely outside the
        viewport. Off by default so recorded data keeps the whole page; the
        `in_viewport` flag says what was visible.

    Returns:
      An observer bound to `page`.
    """
    return Observer(page, include_text=include_text,
                    cull_offscreen=cull_offscreen)
