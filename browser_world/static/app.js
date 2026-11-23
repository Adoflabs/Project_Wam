// Booking page behaviour. All rules live on the server; this file only
// renders server responses and wires up the three-step wizard.
'use strict';

const SESSION =
    new URLSearchParams(window.location.search).get('session') || 'default';
const STEPS = ['appointment', 'details', 'review', 'done'];
const STEP_LABELS = {
  appointment: 'Step 1 of 3',
  details: 'Step 2 of 3',
  review: 'Step 3 of 3',
  done: 'Booking complete',
};
const APPOINTMENT_FIELDS = ['service', 'doctor', 'date', 'slot'];
const DETAIL_FIELDS = ['name', 'email', 'phone', 'consent'];

const $ = (id) => document.getElementById(id);

// Incremented on every slot request so a slow, stale response can never
// overwrite a newer one. This keeps the page deterministic.
let slotRequestId = 0;

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
      free ? `${free} times available.` : 'No times available on this date.';
}

async function loadSlots() {
  const doctor = $('doctor').value;
  const date = $('date').value;
  if (!doctor || !date) {
    clearSlots('Choose a date to see times.');
    return;
  }
  clearSlots('Loading times…');
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

function showStep(step) {
  for (const name of STEPS) {
    $(`step-${name}`).hidden = name !== step;
  }
  $('step-indicator').textContent = STEP_LABELS[step];
  $(`step-${step}`).querySelector('h2').focus();
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

function optionText(id) {
  const select = $(id);
  return select.options[select.selectedIndex].text;
}

function renderReview() {
  const rows = [
    ['Service', optionText('service')],
    ['Doctor', optionText('doctor')],
    ['Date', optionText('date')],
    ['Time', selectedSlot()],
    ['Name', $('name').value.trim()],
    ['Email', $('email').value.trim()],
    ['Phone', $('phone').value.trim()],
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
  clearSelect('doctor', 'Select a doctor');
  clearSelect('date', 'Select a date');
  clearSlots('Choose a date to see times.');
  const service = $('service').value;
  if (!service) {
    return;
  }
  const {data} = await api(`/api/doctors?service=${service}`);
  fillSelect($('doctor'), 'Select a doctor',
             data.map((d) => ({label: d.name, value: d.key})));
}

async function onDoctorChange() {
  clearErrors(['doctor', 'date', 'slot']);
  clearSelect('date', 'Select a date');
  clearSlots('Choose a date to see times.');
  const doctor = $('doctor').value;
  if (!doctor) {
    return;
  }
  const {data} = await api(`/api/dates?doctor=${doctor}`);
  fillSelect($('date'), 'Select a date',
             data.map((d) => ({label: d.label, value: d.iso})));
}

async function onDateChange() {
  clearErrors(['date', 'slot']);
  await loadSlots();
}

async function onAppointmentSubmit(event) {
  event.preventDefault();
  const {data} = await api('/api/validate/appointment', appointmentValues());
  showErrors(APPOINTMENT_FIELDS, data.errors);
  if (Object.keys(data.errors).length === 0) {
    showStep('details');
  }
}

async function onDetailsSubmit(event) {
  event.preventDefault();
  const {data} = await api('/api/validate/details', detailValues());
  showErrors(DETAIL_FIELDS, data.errors);
  if (Object.keys(data.errors).length === 0) {
    clearConfirmError();
    renderReview();
    showStep('review');
  }
}

async function onConfirm() {
  clearConfirmError();
  const {status, data} =
      await api('/api/bookings', {...appointmentValues(), ...detailValues()});
  if (status === 201) {
    $('reference').textContent = `Your reference is ${data.reference}.`;
    showStep('done');
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
  showStep('appointment');
  await loadSlots();
}

async function onBookAnother() {
  $('form-appointment').reset();
  $('form-details').reset();
  clearErrors([...APPOINTMENT_FIELDS, ...DETAIL_FIELDS]);
  clearSelect('doctor', 'Select a doctor');
  clearSelect('date', 'Select a date');
  clearSlots('Choose a date to see times.');
  showStep('appointment');
}

async function init() {
  const {data} = await api('/api/services');
  fillSelect($('service'), 'Select a service',
             data.map((s) => ({label: s.name, value: s.key})));
  $('service').addEventListener('change', onServiceChange);
  $('doctor').addEventListener('change', onDoctorChange);
  $('date').addEventListener('change', onDateChange);
  $('slot-list').addEventListener('change', () => clearErrors(['slot']));
  $('form-appointment').addEventListener('submit', onAppointmentSubmit);
  $('form-details').addEventListener('submit', onDetailsSubmit);
  $('back-details').addEventListener('click', () => showStep('appointment'));
  $('back-review').addEventListener('click', () => showStep('details'));
  $('confirm').addEventListener('click', onConfirm);
  $('choose-another').addEventListener('click', onChooseAnother);
  $('book-another').addEventListener('click', onBookAnother);
  document.body.dataset.ready = 'true';
}

init();
