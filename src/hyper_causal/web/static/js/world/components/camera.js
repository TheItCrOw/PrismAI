import { PerspectiveCamera } from 'https://cdn.skypack.dev/three@0.132.2';
import { TWEEN } from './../systems/tween.js';

function createCamera() {
  const camera = new PerspectiveCamera(
    35, // fov = Field Of View
    1, // aspect ratio (dummy value)
    0.1, // near clipping plane
    1000, // far clipping plane (distance)
  );

  // move the camera back so we can view the scene
  camera.position.set(0, 0, 10);

  return camera;
}

function moveCameraX(camera, newXPosition) {
  var tween = new TWEEN.Tween(camera.position)
    .to({
      x: newXPosition,
      y: camera.position.y,
      z: camera.position.z
    },
      100)
    .easing(TWEEN.Easing.Quadratic.Out)
    .start();

  TWEEN.update();
}

export { createCamera, moveCameraX };
