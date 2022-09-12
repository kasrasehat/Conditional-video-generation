# Conditional-video-generation

### Explain why did you design the video dataloader in this way?
We design data loader in this way in order to inherit many classes of pytorch
which are designed to handle RAM and feed data to model batch by batch. Also, our labels which are captions for each video, has different length which makes creating dataloader 
challenging. if we want to use another type of dataloader we have to pad each label to make length of all strings equal. such work for handling data will lead to 
increasing in the loss. Also, it is crucial to clear cache of gpu ram after each epoch or each iteration for preventing it from being full.

### What are the weaknesses of your video loader?
The main weakness of designed dataloader is this point that it has to read all frames of video in order to 
sample N random frames and it makes sampling process time-consuming specially in long videos and huge datasets.
