import { World } from './world/world.js';

async function main() {
    const container = document.querySelector('#scene-container');

    const world = new World(container);
    await world.init();

    world.start();
}

main().catch((err) => {
    console.error(err);
});
