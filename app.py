import streamlit as st

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Conversational Memory Chatbot",
    page_icon="🤖"
)

st.title("🤖 Conversational Memory Chatbot")

st.write("This chatbot remembers the previous messages in the conversation.")

# ---------------------------------------
# Initialize Chat History
# ---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------
# Display Previous Messages
# ---------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------
# Chat Input
# ---------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    # Save user's message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # ---------------------------------------
    # Simple memory-based response
    # ---------------------------------------
    user_text = user_input.lower()

    if "my name is" in user_text:

        name = user_input.split("my name is", 1)[1].strip()

        st.session_state.user_name = name

        response = f"Nice to meet you, {name}! I will remember your name during this chat."

    elif "what is my name" in user_text:

        if "user_name" in st.session_state:

            response = (
                f"Your name is {st.session_state.user_name}. "
                "I remember it from our earlier conversation."
            )

        else:

            response = "You haven't told me your name yet."

    elif "hello" in user_text or "hi" in user_text:

        if "user_name" in st.session_state:

            response = f"Hello {st.session_state.user_name}! How can I help you?"

        else:

            response = "Hello! How can I help you?"

    else:

        # Count previous messages to demonstrate memory
        message_count = len(st.session_state.messages)

        response = (
            f"I remember our conversation. "
            f"This is message #{message_count} from you."
        )

    # Save chatbot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Display chatbot response
    with st.chat_message("assistant"):
        st.write(response)

# ---------------------------------------
# Clear Conversation
# ---------------------------------------
if st.session_state.messages:

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        if "user_name" in st.session_state:
            del st.session_state.user_name

        st.rerun()

