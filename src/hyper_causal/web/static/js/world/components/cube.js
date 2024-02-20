import {
  BoxBufferGeometry,
  Mesh,
  MathUtils,
  MeshStandardMaterial,
  TextureLoader
} from 'https://cdn.skypack.dev/three@0.132.2';

function createMaterial() {
  // create a texture loader.
  const textureLoader = new TextureLoader();

  // load a texture
  const texture = textureLoader.load(
    'static/img/assets/textures/uv-test-bw.png',
  );

  // create a "standard" material using
  // the texture we just loaded as a color map
  const material = new MeshStandardMaterial({
    map: texture,
    // color: 'red'
  });

  return material;
}

function createCube() {
  const geometry = new BoxBufferGeometry(2, 2, 2);
  const material = createMaterial();
  const cube = new Mesh(geometry, material);

  cube.rotation.set(-0.5, -0.1, 0.8);

  const radiansPerSecond = MathUtils.degToRad(30);

  cube.tick = (delta) => {
    // increase the cube's rotation each frame
    cube.rotation.z += delta * radiansPerSecond;
    cube.rotation.x += delta * radiansPerSecond;
    cube.rotation.y += delta * radiansPerSecond;
  };

  return cube;
}

export { createCube };

