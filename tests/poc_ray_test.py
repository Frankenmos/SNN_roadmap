import ray
import torch

@ray.remote
class DummyActor:
    def __init__(self, actor_id):
        self._actor_id = actor_id

    def get_data(self):
        print(f"Actor {self._actor_id} is generating data.")
        return torch.rand(3, 84, 84)

def main():
    ray.init()
    print("Ray has been initialized.")

    # Create 4 dummy actors
    actors = [DummyActor.remote(i) for i in range(4)]
    print(f"{len(actors)} dummy actors have been created.")

    # Collect data from the actors
    for i in range(10):
        print(f"Learner is collecting data for iteration {i+1}.")
        futures = [actor.get_data.remote() for actor in actors]
        data = ray.get(futures)
        print(f"Learner has collected {len(data)} tensors.")

    ray.shutdown()
    print("Ray has been shut down.")

if __name__ == "__main__":
    main()
