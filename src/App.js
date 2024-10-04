import React, { useState, useEffect, Fragment } from "react";
import * as tf from "@tensorflow/tfjs";
import { DropzoneArea } from "material-ui-dropzone";
import { Backdrop, Chip, CircularProgress, Grid, Stack } from "@mui/material";


class FixedDropout extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.rate = config.rate;
  }

  call(inputs, kwargs) {
    // Implement the dropout operation
    const training = kwargs.training || false;
    if (training) {
      return tf.dropout(inputs[0], this.rate);
    }
    return inputs[0];  // During inference, dropout is not applied.
  }

  static get className() {
    return 'FixedDropout';  // This name should match the one used in the model definition
  }
}


tf.serialization.registerClass(FixedDropout);

// function gelu(x) {
//   return tf.mul(0.5, tf.add(x, tf.mul(x, tf.tanh(tf.mul(tf.sqrt(2 / Math.PI), tf.add(x, tf.mul(0.044715, tf.pow(x, 3))))))));
// }

// Register the GELU activation function so that the model can deserialize it
// class GELU extends tf.layers.Layer {
//   static get className() {
//     return 'gelu';
//   }

//   call(inputs) {
//     const input = inputs[0];
//     return gelu(input);
//   }

//   static fromConfig(config) {
//     return new GELU(config);
//   }
// };

// tf.serialization.registerClass(GELU);

// Register the custom layer with TensorFlow.js
// tf.serialization.registerClass(FixedDropout);


// Define a custom layer to replicate TFOpLambda functionality
// class TFOpLambda extends tf.layers.Layer {
//   constructor(config) {
//     super(config);
//     // Store any custom parameters here if necessary
//   }

//   // Define the behavior of the layer
//   call(inputs, kwargs) {
//     const input = inputs[0]; // Extract the input tensor
//     // Implement the equivalent of the lambda operation here
//     return input;  // Modify this based on the original Lambda function logic
//   }

//   // To properly register the class so it can be loaded
//   static get className() {
//     return 'TFOpLambda';
//   }

//   static fromConfig(config) {
//     return new TFOpLambda(config);
//   }
// }

// // Register the custom TFOpLambda layer so it can be deserialized
// tf.serialization.registerClass(TFOpLambda);



// class HardSigmoidTorch extends tf.layers.Layer {
//   constructor(config) {
//     super(config);
//   }

//   // Define the custom activation function logic
//   call(inputs) {
//     const input = inputs[0];
//     return tf.clipByValue(tf.add(tf.mul(0.2, input), 0.5), 0, 1);  // hardSigmoid logic
//   }

//   static get className() {
//     return 'hardSigmoidTorch';  // Ensure this matches the activation name in the model
//   }

//   static fromConfig(config) {
//     return new HardSigmoidTorch(config);
//   }
// }
// tf.serialization.registerClass(HardSigmoidTorch); 


function App() {
  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      // const model_url = "tfjs/MobileNetV3Large/model.json";
      // const model_url = "MxnetEye_tf241/model.json";
      const model_url = "model.json";

      
      // const model_url = "fastvit_t8_imagenet/model.json";
      // const model_url = "ghostnet_050_imagenet/model.json";

      const model = await tf.loadLayersModel(model_url);
     
      console.log("LoadModelSummary",model.summary());

      setModel(model);
    };

    const getClassLabels = async () => {
      const res = await fetch(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
      );

      const data = await res.json();

      setClassLabels(data);
    };

    loadModel();
    getClassLabels();
  }, []);

  const readImageFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();

      reader.onload = () => resolve(reader.result);

      reader.readAsDataURL(file);
    });
  };

  const createHTMLImageElement = (imageSrc) => {
    return new Promise((resolve) => {
      const img = new Image();

      img.onload = () => resolve(img);

      img.src = imageSrc;
    });
  };

  const handleImageChange = async (files) => {
    if (files.length === 0) {
      setConfidence(null);
      setPredictedClass(null);
    }

    if (files.length === 1) {
      setLoading(true);

      const imageSrc = await readImageFile(files[0]);
      const image = await createHTMLImageElement(imageSrc);

      // tf.tidy for automatic memory cleanup
      const confidence = tf.tidy( () => {
        const tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
    console.log(tensorImg.shape,"===", model.input.shape, "\ntensorImg", tensorImg);


        const normalizedTensor = tensorImg.div(255.0);

        console.log("normalizedTensor", normalizedTensor);
        
        const result = model.predict(normalizedTensor);

        // const dummyInput = tf.ones([1,224,224,3])
        // const result =  model.predict(dummyInput);
        console.log(result, "result");

        const predictions =  result.data().then((prediction)=>{
          console.log("prediction", prediction[0]);
          setConfidence(prediction[0]);
          
        });
        console.log("predictions", predictions);
        const confidence = predictions[0]
        return confidence;
      });

      // setPredictedClass(predictedClass);
      setConfidence(confidence);
      setLoading(false);
    }
  };

  return (
    <Fragment>
      <Grid container className="App" direction="column" alignItems="center" justifyContent="center" marginTop="12%">
        <Grid item>
          <h1 style={{ textAlign: "center", marginBottom: "1.5em" }}>Open Close Eye Detector</h1>
          <DropzoneArea
            acceptedFiles={["image/*"]}
            dropzoneText={"Add an image"}
            onChange={handleImageChange}
            maxFileSize={10000000}
            filesLimit={1}
            showAlerts={["error"]}
          />
          <Stack style={{ marginTop: "2em", width: "12rem" }} direction="row" spacing={1}>
            {/* <Chip
              label={predictedClass === null ? "Prediction:" : `Prediction: ${predictedClass}`}
              style={{ justifyContent: "left" }}
              variant="outlined"
            /> */}

   <Chip
              label={`Prediction: ${confidence}`}
              style={{ justifyContent: "left" }}
              variant="outlined"
            />
          </Stack>
        </Grid>
      </Grid>

      <Backdrop sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }} open={loading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </Fragment>
  );
}

export default App;
