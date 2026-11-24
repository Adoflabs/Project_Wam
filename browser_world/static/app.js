// Booking page behaviour. All rules live on the server; this file renders
// server responses and runs the steps in the order the scenario's layout
// asks for, with the scenario's wording.
'use strict';

const SESSION =
    new URLSearchParams(window.location.search).get('session') || 'default';
// Each layout is a list of steps; each step shows one or more sections.
const LAYOUTS = {
  wizard: [['appointment'], ['details'], ['review']],
  details_first: [['details'], ['appointment'], ['review']],
  combined: [['appointment', 'details'], ['review']],
};
const SECTIONS = ['appointment', 'details', 'review', 'done'];
const APPOINTMENT_FIELDS = ['service', 'doctor', 'date', 'slot'];
const DETAIL_FIELDS = ['name', 'email', 'phone', 'consent'];
const FIELDS_OF = {appointment: APPOINTMENT_FIELDS, details: DETAIL_FIELDS};
const BACK_BUTTON_OF = {appointment: 'back-appointment',
                        details: 'back-details'};

const $ = (id) => document.getElementById(id);
const byKey = (key) => document.querySelector(`[data-wam-key="${key}"]`);

let text = {};
let steps = LAYOUTS.wizard;
let current = 0;

// Incremented on every slot request so a slow, stale response can never
// overwrite a newer one. This keeps the page deterministic.
let slotRequestId = 0;

function format(template, values) {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
}

async function api(path, body) {
  const options = {headers: {'X-WAM-Session': SESSION}};
  if (body !== undefined) {
    options.method = 'POST';
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  return {status: response.status, data: await response.json()};
}

function applyText() {
  document.title = text.title;
  document.querySelector('h1').textContent = text.title;
  document.querySelector('header').textContent = text.clinic;
  for (const section of SECTIONS) {
    $(`step-${section}`).querySelector('h2').textContent =
        text[`h_${section}`];
  }
  for (const field of ['service', 'doctor', 'date', 'name', 'email', 'phone',
                       'notes', 'consent']) {
    document.querySelector(`label[for="${field}"]`).textContent = text[field];
  }
  $('slot-group').querySelector('legend').textContent = text.time;
  for (const key of ['next-appointment', 'next-details']) {
    byKey(key).textContent = text.next;
  }
  for (const key of ['back-appointment', 'back-details', 'back-review']) {
    byKey(key).textContent = text.back;
  }
  byKey('confirm').textContent = text.confirm;
  byKey('choose-another').textContent = text.choose_another;
  byKey('book-another').textContent = text.book_another;
}

function fillSelect(select, placeholder, items) {
  select.replaceChildren(new Option(placeholder, ''));
  for (const item of items) {
    select.append(new Option(item.label, item.value));
  }
  select.disabled = items.length === 0;
}

function clearSelect(id, placeholder) {
  fillSelect($(id), placeholder, []);
}

function clearSlots(message) {
  slotRequestId += 1;
  $('slot-list').replaceChildren();
  $('slot-status').textContent = message;
}

function selectedSlot() {
  const checked = document.querySelector('input[name=slot]:checked');
  return checked ? checked.value : '';
}

function renderSlots(slots) {
  const list = $('slot-list');
  list.replaceChildren();
  for (const slot of slots) {
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'slot';
    input.value = slot.time;
    input.disabled = !slot.available;
    input.dataset.wamKey = `slot-${slot.time}`;
    const label = document.createElement('label');
    label.append(input, ` ${slot.time}`);
    list.append(label);
  }
  const free = slots.filter((s) => s.available).length;
  $('slot-status').textContent =
      free ? format(text.times_free, {n: free}) : text.times_none;
}

async function loadSlots() {
  const doctor = $('doctor').value;
  const date = $('date').value;
  if (!doctor || !date) {
    clearSlots(text.times_prompt);
    return;
  }
  clearSlots(text.times_loading);
  const requestId = slotRequestId;
  const query = new URLSearchParams({doctor, date});
  const {data} = await api(`/api/slots?${query}`);
  if (requestId === slotRequestId) {
    renderSlots(data);
  }
}

function setError(field, message) {
  const error = $(`${field}-error`);
  const control = field === 'slot' ? $('slot-group') : $(field);
  error.textContent = message;
  error.hidden = false;
  control.setAttribute('aria-invalid', 'true');
}

function clearErrors(fields) {
  for (const field of fields) {
    const error = $(`${field}-error`);
    const control = field === 'slot' ? $('slot-group') : $(field);
    error.textContent = '';
    error.hidden = true;
    control.removeAttribute('aria-invalid');
  }
}

function showErrors(fields, errors) {
  clearErrors(fields);
  for (const field of fields) {
    if (errors[field]) {
      setError(field, errors[field]);
    }
  }
}

function showStep(index, focus = true) {
  current = index;
  const shown = steps[index];
  for (const section of SECTIONS) {
    $(`step-${section}`).hidden = !shown.includes(section);
  }
  // Only the last section of a later step offers Back.
  for (const [section, id] of Object.entries(BACK_BUTTON_OF)) {
    $(id).hidden = index === 0 || section !== shown[shown.length - 1];
  }
  // A combined step is submitted once, from its last section.
  byKey('next-appointment').hidden =
      shown.includes('details') && shown.includes('appointment');
  $('step-indicator').textContent =
      format(text.step, {i: index + 1, n: steps.length});
  if (focus) {
    $(`step-${shown[0]}`).querySelector('h2').focus();
  }
}

function showDone() {
  for (const section of SECTIONS) {
    $(`step-${section}`).hidden = section !== 'done';
  }
  $('step-indicator').textContent = text.done;
  $('step-done').querySelector('h2').focus();
}

function appointmentValues() {
  return {
    service: $('service').value,
    doctor: $('doctor').value,
    date: $('date').value,
    slot: selectedSlot(),
  };
}

function detailValues() {
  return {
    name: $('name').value,
    email: $('email').value,
    phone: $('phone').value,
    consent: $('consent').checked ? 'true' : 'false',
  };
}

const VALUES_OF = {appointment: appointmentValues, details: detailValues};

function optionText(id) {
  const select = $(id);
  return select.options[select.selectedIndex].text;
}

function renderReview() {
  const rows = [
    [text.service, optionText('service')],
    [text.doctor, optionText('doctor')],
    [text.date, optionText('date')],
    [text.time, selectedSlot()],
    [text.name, $('name').value.trim()],
    [text.email, $('email').value.trim()],
    [text.phone, $('phone').value.trim()],
  ];
  $('review-list').replaceChildren(...rows.map(([label, value]) => {
    const item = document.createElement('li');
    item.textContent = `${label}: ${value}`;
    return item;
  }));
}

function clearConfirmError() {
  $('confirm-error').textContent = '';
  $('confirm-error').hidden = true;
  $('choose-another').hidden = true;
  $('confirm').hidden = false;
}

async function onServiceChange() {
  clearErrors(APPOINTMENT_FIELDS);
  clearSelect('doctor', text.pick_doctor);
  clearSelect('date', text.pick_date);
  clearSlots(text.times_prompt);
  const service = $('service').value;
  if (!service) {
    return;
  }
  const {data} = await api(`/api/doctors?service=${service}`);
  fillSelect($('doctor'), text.pick_doctor,
             data.map((d) => ({label: d.name, value: d.key})));
}

async function onDoctorChange() {
  clearErrors(['doctor', 'date', 'slot']);
  clearSelect('date', text.pick_date);
  clearSlots(text.times_prompt);
  const doctor = $('doctor').value;
  if (!doctor) {
    return;
  }
  const {data} = await api(`/api/dates?doctor=${doctor}`);
  fillSelect($('date'), text.pick_date,
             data.map((d) => ({label: d.label, value: d.iso})));
}

async function onDateChange() {
  clearErrors(['date', 'slot']);
  await loadSlots();
}

// Validates every section of the current step on the server, then moves on.
async function onSubmit(event) {
  event.preventDefault();
  let valid = true;
  for (const section of steps[current]) {
    const {data} = await api(`/api/validate/${section}`, VALUES_OF[section]());
    showErrors(FIELDS_OF[section], data.errors);
    valid = valid && Object.keys(data.errors).length === 0;
  }
  if (!valid) {
    return;
  }
  if (steps[current + 1].includes('review')) {
    clearConfirmError();
    renderReview();
  }
  showStep(current + 1);
}

async function onConfirm() {
  clearConfirmError();
  const {status, data} =
      await api('/api/bookings', {...appointmentValues(), ...detailValues()});
  if (status === 201) {
    $('reference').textContent = format(text.reference, {ref: data.reference});
    showDone();
    return;
  }
  const message = status === 409 ?
      data.error : Object.values(data.errors || {})[0] || 'Booking failed.';
  $('confirm-error').textContent = message;
  $('confirm-error').hidden = false;
  if (status === 409) {
    $('choose-another').hidden = false;
    $('confirm').hidden = true;
  }
}

async function onChooseAnother() {
  clearConfirmError();
  showStep(steps.findIndex((step) => step.includes('appointment')));
  await loadSlots();
}

async function onBookAnother() {
  $('form-appointment').reset();
  $('form-details').reset();
  clearErrors([...APPOINTMENT_FIELDS, ...DETAIL_FIELDS]);
  clearSelect('doctor', text.pick_doctor);
  clearSelect('date', text.pick_date);
  clearSlots(text.times_prompt);
  showStep(0);
}

async function init() {
  const presentation = (await api('/api/presentation')).data;
  text = presentation.text;
  steps = LAYOUTS[presentation.layout];
  document.body.classList.add(`layout-${presentation.layout}`);
  applyText();
  const {data} = await api('/api/services');
  fillSelect($('service'), text.pick_service,
             data.map((s) => ({label: s.name, value: s.key})));
  clearSelect('doctor', text.pick_doctor);
  clearSelect('date', text.pick_date);
  clearSlots(text.times_prompt);
  $('service').addEventListener('change', onServiceChange);
  $('doctor').addEventListener('change', onDoctorChange);
  $('date').addEventListener('change', onDateChange);
  $('slot-list').addEventListener('change', () => clearErrors(['slot']));
  $('form-appointment').addEventListener('submit', onSubmit);
  $('form-details').addEventListener('submit', onSubmit);
  for (const id of ['back-appointment', 'back-details', 'back-review']) {
    $(id).addEventListener('click', () => showStep(current - 1));
  }
  $('confirm').addEventListener('click', onConfirm);
  $('choose-another').addEventListener('click', onChooseAnother);
  $('book-another').addEventListener('click', onBookAnother);
  showStep(0, false);
  document.body.dataset.ready = 'true';
}

init();
