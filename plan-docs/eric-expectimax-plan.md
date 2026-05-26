# Depth Limited Expectimax (Why)

## Compute (Claims)
- In terms of compute, this one is highly configurable, and changing the depth to fit the time limits should be trivial.

- Its important to note that iterative deepening is also a possible option here, and may be more helpful for debugging and time-aware action computation if this approach **frequently and unpredictably** times out

- Depth-limiting the search can be implicitly defined using the ply count in the constructor of the MinimaxLikeAgent. Note that the ply_count represents the ***number of full turns that go back to the playing agent***.
    - This means 4 (including the playing agent) if we consider both enemies and the teamate. I think it may be easier to not-consider the teamate, as it lets us potentially push the `ply_count` to 2, and for now is more simple to implement.
    - Note that the `ply_count` of 1 or 2 makes this implementation more of a reflex agent with some lookahead.
    - See the usage of `ply_count` in depth-limiting [here](../pacai/agents/minimax.py#L108-130)
    - Skipping the teamate agent requires the `minimax_step` function to go to the next agent index when the current agent index is the non-playing teamate.

- Alpha-beta pruning can't really happen on expectimax, but there is a method that helps create similar pruning behavior [described here](https://en.wikipedia.org/wiki/Expectiminimax) (it's wicked though and require upper-lower bounds for chance nodes, which may not exist in our case)



### DistancePreComputer
- This is the most important setup for this to work effectively, all of the action selection time should be dedicated to the expectimax search, NOT to computing distances for eval functions. This is a necessary aspect of the implementation.

- A quick look into what this would look like:
    - The distance pre-computer allows an easy setup as long as a board is provided
    - Thus the base Agent class gives us a [self.extra_storage](../pacai/core/agent.py#L84) that allows us to store arbitrary fields
    - We can override the existing [game_start_full() or game_start()](../pacai/core/agent.py#L121-139) in the base agent class to initialize the pre-computer

    ```c++
    distance_precompute = DistancePreComputer()
    distance_precompute.compute(initial_state.board)
    self.extra_storage["distancePrecompute"] = distance_precompute
    ``` 

### Time Testing

This step of the process should be relatively easy. We should take multiple different boards, preferrably larger ones like `capture-default`, `capture-jumbo`, and different random boards against multiple baseline agents. For this, it's best to just wrap our initial setup and `get_action` methods with a  time counter, and then to use the logger to get this information, which may be a bit faster than using print.
-  Must also assume whatever time we get on our laptops is way faster than the class server. Likely best to go based on p99 times.

***


## Existing Code Structure and Modifications

Currently, the `MinimaxLikeAgent` in [minimax.py](../pacai/agents/minimax.py) supports setting up expectimax via a boolean flag in the constructor.

See reference code here:
```python
def __init__(self,
            ply_count: int = DEFAULT_PLY_COUNT,
            alphabeta_prune: bool = False,
            expectimax: bool = False,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # Parse (possibly string) arguments.
        ply_count = int(ply_count)
        alphabeta_prune = pacai.util.parse.boolean(alphabeta_prune)
        expectimax = pacai.util.parse.boolean(expectimax)
```

From then on, the rest of our base Expectimax implementation is pretty basic. All that needs implementation is overriding the `minimax_step_max`, `minimax_step_expected_min`. `minimax_step_min` is not used here.

### Offense vs Defense: Multiple Roles from One Class

Implementing our `ExpectimaxAgent` wrapper over the `MinimaxLikeAgent` and letting it be abstracted to allow multiple different roles implies two main aspects. This could allow both submitted agents to be our `ExpectimaxAgent` by just taking in different feature and weight parameters.

Setup for the evaluation function for leaf nodes is important, as this is still mostly a reflex agent. 

1. **A feature extractor function**. We can actually see an example in the code for PA3 question six. All this requires is that we let the class take in a feature extractor function as a parameter. Allows us to inject different behavior for what factors matter based on the agents *'personality'* or whether its on offense or defense.
    - Highly important that this fn takes in the `DistancePreComputer` for distance based features.
    - Usage of this feature extractor in a simplified evaluation function may look like below. The one caveat is that we may need to setup a customer getter for self.weights that gates the correct value given the agent's intended behavior and state.
    ```python
    def eval_func(self, game_state: GameState, precomputer: DistancePreComputer) -> float:
        features = self.feature_extraction_func(self, game_state, precomputer)
        eval = 0
        for f, v in features.items():
            eval += v * self.weights[f]
        return eval
    ```

2. A configurable preloaded dictionary of weights in the constructor for the class. We do have to confirm that the set of weights that we pass into the constructor also has the same shape as the features returned from the feature extraction function. 
    - Thus, we need to consider the different weights that will be used whether the agent is on offense or defense, or based on the general behavior of our agent. 

***

### Crossing the Center of the Board

Handling transitions over the ply_count of the Expectimax search doesn't need to be *explicitly handled*. The evaluation function should be agnostic to which side of the board the agent is on. 

However, the feature extractor and the weights getter method do care, and **this is where these state dependent features and weights will be abstracted away to**. If we decide that we care about the agent crossing over during the search, we can use its last position to include that as a boolean flag feature with its own weight.

***

## Learning Weights (Ehhh prety optimistic)

In order to make tuning the weights faster, and potentially more optimal we can setup some pretty simple training loops in the test environment.

1. Keep feature weights as brought in from `.json` files in the repo. With ranges for its values to allow a grid search approach.
    - We will probably have to also specify the order to iterate over the weights in the grid search. Either way we have to be careful, because this grid will be massive and require many (20+) runs for each combination of weights.

```json
{
    feature_label: "nearest_food_distance",
    range: [0, 50],
    step: 0.5  // step size to take in range
    current_value: 24.5 // optional, may help if grid cant complete in one run
}
```

2. During a grid search run, either do the whole thing one go, whole grid in memory, or have each run log out its recently used weight (see the current_value) field. The next time the agent picks up the feature, it will start from a `step` above the `current_value`. (To be elaborated)
3. Consider grid-search runs across multiple setups, different boards, baseline agents, etc. Hold out a few setups for a test-set.