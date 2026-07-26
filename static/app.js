const form = document.getElementById('chat-form');
const input = document.getElementById('message-input');
const messages = document.getElementById('messages');
const hint = document.getElementById('hint');

function addMessage(text, type) {
  const bubble = document.createElement('div');
  bubble.className = `message ${type}`;
  bubble.textContent = text;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage(message) {
  const response = await fetch('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error('Request failed');
  }

  return response.json();
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const message = input.value.trim();
  if (!message) {
    hint.textContent = 'Type a question before sending.';
    return;
  }

  addMessage(message, 'user');
  input.value = '';
  input.focus();
  hint.textContent = 'Thinking...';

  try {
    const data = await sendMessage(message);
    addMessage(data.reply, 'bot');
    hint.textContent = `Confidence: ${data.confidence} | Threshold: ${data.threshold}`;
  } catch (error) {
    addMessage('Something went wrong while getting a reply.', 'bot');
    hint.textContent = 'Try again in a moment.';
  }
});

input.focus();
