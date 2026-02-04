const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const main = document.getElementById("wrap");

messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
});

sendButton.addEventListener('click', () => {
    const user_message = messageInput.value;
    if (user_message) {
        const messageElement = document.createElement('div');
        messageElement.textContent = user_message;
        messageElement.classList.add('user-message-div');

        main.appendChild(messageElement);

        console.log("전송된 메시지:", user_message); // 실제 전송 기능 구현

        // 메시지 입력창 초기화
        messageInput.value = '';
        messageInput.style.height = 'auto'; // 높이 초기화
    }
});
