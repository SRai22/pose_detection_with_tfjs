import * as tf from '@tensorflow/tfjs-core';
import {showBackendConfigs} from './option_panel';
import * as params from './params';
import * as posedetection from '@tensorflow-models/pose-detection';

export function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Reset the target backend.
 *
 * @param backendName The name of the backend to be reset.
 */
async function resetBackend(backendName) {
  const ENGINE = tf.engine();
  if (!(backendName in ENGINE.registryFactory)) {
      if(backendName === 'webgpu') {
          alert('webgpu backend is not registered. Your browser may not support WebGPU yet. To test this backend, please use a supported browser, e.g. Chrome canary with --enable-unsafe-webgpu flag');
          params.STATE.backend = !!params.STATE.lastTFJSBackend ? params.STATE.lastTFJSBackend : 'tfjs-webgl';
          showBackendConfigs();
          return;
      } else {
      throw new Error(`${backendName} backend is not registered.`);
      }
  }

  if (backendName in ENGINE.registry) {
      const backendFactory = tf.findBackendFactory(backendName);
      tf.removeBackend(backendName);
      tf.registerBackend(backendName, backendFactory);
  }

  await tf.setBackend(backendName);
  params.STATE.lastTFJSBackend = `tfjs-${backendName}`;
}

/**
* Set environment flags.
*
* This is a wrapper function of `tf.env().setFlags()` to constrain users to
* only set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`).
*
* ```js
* const flagConfig = {
*        WEBGL_PACK: false,
*      };
* await setEnvFlags(flagConfig);
*
* console.log(tf.env().getBool('WEBGL_PACK')); // false
* console.log(tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')); // false
* ```
*
* @param flagConfig An object to store flag-value pairs.
*/
export async function setBackendAndEnvFlags(flagConfig, backend) {
  if (flagConfig == null) {
      return;
  } else if (typeof flagConfig !== 'object') {
      throw new Error(
          `An object is expected, while a(n) ${typeof flagConfig} is found.`);
  }

  // Check the validation of flags and values.
  for (const flag in flagConfig) {
      if (!(flag in params.TUNABLE_FLAG_VALUE_RANGE_MAP)) {
      throw new Error(`${flag} is not a tunable or valid environment flag.`);
      }
      if (params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag].indexOf(flagConfig[flag]) === -1) {
      throw new Error(
          `${flag} value is expected to be in the range [${
              params.TUNABLE_FLAG_VALUE_RANGE_MAP[flag]}], while ${flagConfig[flag]}` +
          ' is found.');
      }
  }

  tf.env().setFlags(flagConfig);

  const [runtime, $backend] = backend.split('-');

  if (runtime === 'tfjs') {
      await resetBackend($backend);
  }
}

// These anchor points allow the pose pointcloud to resize according to its
// position in the input.
const ANCHOR_POINTS = [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, -1, 0]];

// #ffffff - White
// #800000 - Maroon
// #469990 - Malachite
// #e6194b - Crimson
// #42d4f4 - Picton Blue
// #fabed4 - Cupid
// #aaffc3 - Mint Green
// #9a6324 - Kumera
// #000075 - Navy Blue
// #f58231 - Jaffa
// #4363d8 - Royal Blue
// #ffd8b1 - Caramel
// #dcbeff - Mauve
// #808000 - Olive
// #ffe119 - Candlelight
// #911eb4 - Seance
// #bfef45 - Inchworm
// #f032e6 - Razzle Dazzle Rose
// #3cb44b - Chateau Green
// #a9a9a9 - Silver Chalice
const COLOR_PALETTE = [
  '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
  '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
  '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
];

/**
 * Draw the keypoints and skeleton on the video.
 * @param poses A list of poses to render.
 */
export async function drawResults(ctx,poses) {
  for (const pose of poses) {
      drawResult(ctx,pose);
  }
}
/**
* Draw the keypoints and skeleton on the video.
* @param pose A pose with keypoints to render.
*/
async function drawResult(ctx,pose) {
  if (pose.keypoints != null) {
    drawKeypoints(ctx, pose.keypoints);
    drawSkeleton(ctx, pose.keypoints, pose.id);
  }
}

/**
* Draw the keypoints on the video.
* @param keypoints A list of keypoints.
*/
async function drawKeypoints(ctx, keypoints) {
  const keypointInd =
      posedetection.util.getKeypointIndexBySide(params.STATE.model);
  ctx.fillStyle = 'Red';
  ctx.strokeStyle = 'White';
  ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

  for (const i of keypointInd.middle) {
      drawKeypoint(ctx,keypoints[i]);
  }

  ctx.fillStyle = 'Green';
  for (const i of keypointInd.left) {
      drawKeypoint(ctx, keypoints[i]);
  }

  ctx.fillStyle = 'Orange';
  for (const i of keypointInd.right) {
      drawKeypoint(ctx, keypoints[i]);
  }
}

async function drawKeypoint(ctx, keypoint) {
  // If score is null, just show the keypoint.
  const score = keypoint.score != null ? keypoint.score : 0;
  const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

  if (score >= scoreThreshold) {
    const circle = new Path2D();
    circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
    ctx.fill(circle);
    ctx.stroke(circle);
  }
}

/**
* Draw the skeleton of a body on the video.
* @param keypoints A list of keypoints.
*/
async function drawSkeleton(ctx, keypoints, poseId) {
  // Each poseId is mapped to a color in the color palette.
  const color = params.STATE.modelConfig.enableTracking && poseId != null ?
      COLOR_PALETTE[poseId % 20] :
      'Red';
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

  posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([
                                                                    i, j
                                                                  ]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    // If score is null, just show the keypoint.
    const score1 = kp1.score != null ? kp1.score : 1;
    const score2 = kp2.score != null ? kp2.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  });
}