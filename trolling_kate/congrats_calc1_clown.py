import streamlit as st
import time

st.set_page_config(page_title="🎉 Congrats on Calc 1! 🎉", page_icon="🤡", layout="centered")

# Festive header
st.markdown("""
# 🎉🎈 Congratulations! 🎈🎉
## You finished Calc 1!
""")

st.markdown("""
<div style='text-align: center; font-size: 2em;'>
    <b>We're so proud of you!</b>
</div>
""", unsafe_allow_html=True)

# Clown emoji and confetti
st.markdown("""
<div style='text-align: center; font-size: 4em;'>
    🤡
</div>
""", unsafe_allow_html=True)

st.balloons()
st.snow()

# Draw a clown face with Streamlit's canvas (using st.pyplot)
import matplotlib.pyplot as plt
import numpy as np

def draw_clown():
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal')
    ax.axis('off')
    # Face
    face = plt.Circle((0, 0), 1, color='#ffe066', ec='orange', lw=4)
    ax.add_patch(face)
    # Eyes
    ax.add_patch(plt.Circle((-0.4, 0.4), 0.18, color='white', ec='black', lw=2))
    ax.add_patch(plt.Circle((0.4, 0.4), 0.18, color='white', ec='black', lw=2))
    ax.add_patch(plt.Circle((-0.4, 0.4), 0.08, color='blue'))
    ax.add_patch(plt.Circle((0.4, 0.4), 0.08, color='blue'))
    # Nose
    ax.add_patch(plt.Circle((0, 0), 0.18, color='red', ec='black', lw=2))
    # Mouth
    mouth_x = np.linspace(-0.5, 0.5, 100)
    mouth_y = -0.5 * np.sin(np.pi * mouth_x)
    ax.plot(mouth_x, mouth_y - 0.2, color='red', lw=4)
    # Cheeks
    ax.add_patch(plt.Circle((-0.6, -0.2), 0.13, color='pink', alpha=0.7))
    ax.add_patch(plt.Circle((0.6, -0.2), 0.13, color='pink', alpha=0.7))
    # Hat
    hat_x = np.array([-0.5, 0, 0.5])
    hat_y = np.array([0.9, 1.5, 0.9])
    ax.fill(hat_x, hat_y, color='#6fa8dc', ec='navy', lw=2)
    ax.add_patch(plt.Circle((0, 1.5), 0.12, color='red', ec='black', lw=2))
    st.pyplot(fig)

# Button to show clown drawing
if st.button("Show me a clown! 🤡"):
    st.markdown("<div style='text-align: center; font-size: 1.5em;'>Here's a clown just for you! 🎪</div>", unsafe_allow_html=True)
    draw_clown()
    
    # Add camera functionality
    st.write("📸 Now let's see the REAL clown! Take a selfie:")
    img = st.camera_input("Smile for the camera! 🤡")
    
    if img is not None:
        st.write("🎉 Look at you! You're the real star! ⭐")
        st.image(img, caption="The REAL clown! 🤡", use_column_width=True)
        
        # Add more festive effects
        st.balloons()
        st.snow()
        
        st.markdown("""
        <div style='text-align: center; font-size: 1.5em; color: purple;'>
            <b>🎉 Congratulations on finishing Calc 1! 🎉</b><br>
            <b>You're a mathematical superstar! ⭐</b>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1)
    st.success("Keep clowning around with math! 🎉")

st.markdown("""
---
<div style='text-align: center;'>
    <i>From your proud sibling and the clowns of Streamlit! 🤡🎉</i>
</div>
""", unsafe_allow_html=True) 