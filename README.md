# streamlit-experimentation
Repo for testing out different streamlit application ideas

# repo contents:
- st-augustine-fl: sample app that runs an object detector on a live video stream of a street in St. Augustine, FL (https://www.youtube.com/watch?v=YLSELFy-iHQ). NOTE: code is still pretty messy :) 
- nantucket: sample app that provides a "AI-Powered Smart City Dashboard" for a city operator. This app includes the code required to access a live video feed, make API calls with Modzy's edge client, and aggregate the predictions into some interesting insights   


# Usage

### St augustine repo:

1. `cd st-augustine-fl`
2. pip install the requirements file (may be some extra reqs in there, i just created that with a `pip freeze` and didn't strip it of unused dependencies)
3. Make sure you have an instance of modzy core running. Here are the device groups I've connected (model ids and versions are already coded into `app.py`)

        - dev: https://dev.modzy.engineering/operations/edge/groups/device-group-2UIq6Yb0HPdK7c5AHktuXwnVXs8/models

        - demo: https://demo.modzy.engineering/operations/edge/groups/device-group-2U1vIU5lyV3JRxAFMhqHBvsuLB5/devices
        
4. run the app with the streamlit command: `streamlit run app.py`  


### Nantucket Smart Cities

Same as St Augustine app