
import textarena as ta
 
model_name = "loose guy"
model_description = "just for exp"
email = "sahil21@u.nus.edu"


# Initialize agent
agent = ta.agents.AnthropicAgent(model_name="claude-3-5-sonnet-20241022")


env = ta.make_online(
    env_id=["SpellingBee-v0", "SimpleNegotiation-v0", "Poker-v0"], 
    model_name=model_name,
    model_description=model_description,
    email=email
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)


rewards = env.close()