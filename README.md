# RL_IssacGym_Humanoid
clone of github.com/will-d-chen/IsaacGym-RL-Humanoid


I believe the most impactful modifications are related to adjusting the knee joint angles and degrees of freedom. Under the original parameter conditions, the robotic knee joint movements appear rather rigid and unusual. While casually browsing through other humanoid robot code repositories, I noticed that they have tailored the degrees of freedom for each joint based on the requirements of different movements. Considering the unique demands of the forward motion task, I granted the knee joint greater freedom of movement, and fortunately, this led to significantly improved results.

Furthermore, I'm uncertain whether altering such non-hyperparameters adheres to the established norms. If such modifications are considered unconventional, I kindly ask for a bit of leniency and understanding from the community.
