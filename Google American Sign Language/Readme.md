# Goal of the Competition
The goal of this competition is to detect and translate American Sign Language (ASL) fingerspelling into text. You will create a model trained on the largest dataset of its kind, released specifically for this competition. The data includes more than three million fingerspelled characters produced by over 100 Deaf signers captured via the selfie camera of a smartphone with a variety of backgrounds and lighting conditions.

Your work may help move sign language recognition forward, making AI more accessible for the Deaf and Hard of Hearing community.

<img src = "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiaHD3yzznEbLuhZQ0u9xCJfu6X7CZtDgJy3Fdd3oIAt0uXXrs6wa3fFVp8kLRxFLgfKH0joBLgHi-_ykI7An2gmjLFnjwOEzjMmzX5NDUReJgv4EUnbjBqKsJXHe8TD7gylvW7qSSt58hwUFS9KicowMUo8yKGCaBJG2sFS1-Ol-uPn92JyYQYCmk6/s1600/GDS_ASL_FingerspellingCompetition_Banners_BlogImage.png">

# Dataset Description
The goal of this competition is to detect and translate American Sign Language (ASL) fingerspelling into text.

This competition requires submissions to be made in the form of TensorFlow Lite models. You are welcome to train your model using the framework of your choice as long as you convert the model checkpoint into the tflite format prior to submission. Please see the evaluation page for details.

# Files
|______|______|
|---|---|
|**Train**
|path|The path to the landmark file.
|file_id|A unique identifier for the data file.
|participant_id|A unique identifier for the data contributor.
|sequence_id|A unique identifier for the landmark sequence. Each data file may contain many sequences.
|phrase|The labels for the landmark sequence. The train and test datasets contain randomly generated addresses, phone numbers, and urls derived from components of real addresses/phone numbers/urls.
The landmark files contain the same data as in the ASL Signs competition (minus the row ID column) but reshaped into a wide format. This allows you to take advantage of the Parquet format to entirely skip loading landmarks that you aren't using.
|**Landmark**
|sequence_id|A unique identifier for the landmark sequence. Most landmark files contain 1,000 sequences. The sequence ID is used as the dataframe index.
|frame|The frame number within a landmark sequence.
|[x/y/z]_[type]_[landmark_index]|There are now $1,629$ spatial coordinate columns for the $x$, $y$ and $z$ coordinates for each of the $543$ landmarks. The type of landmark is one of `['face', 'left_hand', 'pose', 'right_hand']`. Details of the hand landmark locations can be found here. The spatial coordinates have already been normalized by MediaPipe. Note that the MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values. The landmarks have been converted to float32.
