"""Tests for the four-plane observer against the live booking page."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import json

from browser_world import observer


def test_semantic_plane(world):
    observation = world.watcher.observe()
    service = observation.find('service')
    assert service.role == 'combobox'
    assert service.name == 'Service'
    assert service.value == 'Select a service'
    assert service.states['required']
    assert not service.states['invalid']  # Untouched fields are not invalid.
    assert observation.find('doctor').states['disabled']
    assert observation.find('step-indicator').name == 'Step 1 of 3'


def test_hidden_steps_are_not_observed(world):
    observation = world.watcher.observe()
    assert observation.find('name') is None
    assert observation.find('confirm') is None
    assert observation.find('error-service') is None  # Hidden until shown.


def test_parent_links_follow_accessibility_tree(world):
    observation = world.watcher.observe()
    service = observation.find('service')
    options = [e for e in observation.elements
               if e.parent_id == service.node_id]
    assert [o.name for o in options][:2] == ['Select a service',
                                            'General check-up']
    assert all(o.role == 'option' for o in options)


def test_geometry_plane(world):
    observation = world.watcher.observe()
    service = observation.find('service')
    doctor = observation.find('doctor')
    assert service.box.width > 0 and service.box.height > 0
    assert doctor.box.y > service.box.y  # Doctor is below service.
    assert service.in_viewport
    assert service.hit_testable is True
    option = next(e for e in observation.elements if e.role == 'option')
    assert option.box is None  # Closed dropdown options have no layout.


def test_error_message_moves_later_elements(world):
    before = world.watcher.observe()
    after = world.click('next-appointment')  # Nothing chosen yet.
    assert after.find('error-service') is not None
    assert after.find('doctor').box.y > before.find('doctor').box.y


def test_occlusion_is_detected(world):
    before = world.watcher.observe()
    assert before.find('next-appointment').hit_testable is True
    world.page.evaluate("""() => {
      const overlay = document.createElement('div');
      overlay.style.cssText =
          'position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:99';
      document.body.append(overlay);
    }""")
    after = world.watcher.observe()
    assert after.find('next-appointment').hit_testable is False
    assert after.find('next-appointment').states['disabled'] is False


def test_motion_plane_pinned_header_and_scroll(make_world):
    world = make_world(viewport_height=300)
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', slot=None)
    top = world.watcher.observe()
    banner = next(e for e in top.elements if e.role == 'banner')
    assert banner.pinned
    assert banner.name == 'Riverside Clinic'
    assert not top.find('next-appointment').in_viewport
    assert top.find('next-appointment').hit_testable is None

    world.page.mouse.wheel(0, 400)
    world.watcher.wait_until_settled()
    scrolled = world.watcher.observe()
    assert scrolled.motion.scroll_y > 0
    assert scrolled.find('next-appointment').in_viewport
    assert scrolled.find('next-appointment').hit_testable is True
    assert not scrolled.find('service').pinned


def test_cull_offscreen_drops_invisible_elements(make_world):
    world = make_world(viewport_height=300)
    culling = observer.attach(world.page, cull_offscreen=True)
    observation = culling.observe()
    assert observation.find('next-appointment') is None
    assert observation.find('service') is not None


def test_temporal_counts_reset_after_each_observation(world):
    world.page.select_option('#service', 'checkup')
    world.watcher.wait_until_settled()
    first = world.watcher.observe()
    assert first.temporal.mutations > 0
    assert first.temporal.requests_started == 1
    assert first.temporal.settled is True
    second = world.watcher.observe()
    assert second.temporal.mutations == 0
    assert second.temporal.requests_started == 0
    assert second.temporal.settled is None


def test_rerendered_elements_get_new_node_ids_but_keep_match_keys(world):
    first = world.choose_appointment('checkup', 'dr-patel', '2026-11-24',
                                     slot=None)
    second = world.act(lambda: world.page.select_option('#date',
                                                        '2026-11-25'))
    world.act(lambda: world.page.select_option('#date', '2026-11-24'))
    third = world.watcher.observe()
    del second  # Only the round trip matters.
    old = first.find('slot-09:00')
    new = third.find('slot-09:00')
    assert old.node_id != new.node_id
    assert old.name == new.name


def test_plane_filtering_and_json(world):
    observation = world.watcher.observe()
    semantic_only = observation.to_dict(planes=[observer.SEMANTIC])
    element = semantic_only['elements'][0]
    assert 'role' in element and 'box' not in element
    assert 'motion' not in semantic_only and 'temporal' not in semantic_only
    full = json.loads(json.dumps(observation.to_dict()))
    assert {'motion', 'temporal'} <= set(full)
    assert 'pinned' in full['elements'][0]
    text = observation.to_text()
    assert 'combobox "Service"' in text
    assert text.splitlines()[-1].startswith('TEMPORAL')
