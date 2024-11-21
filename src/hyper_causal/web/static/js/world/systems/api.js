
/*
 * Gets the next token branches from a given input
 * @param {} input 
 * @param {*} callback 
 */
async function getNextTokenBranches(input, k, temp, p, beamWidth, decodingStrategy, llm) {
    const result = await $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: '/api/tokens/next',
        data: JSON.stringify({ input, k, temp, p, beamWidth, decodingStrategy, llm }),
        error: function (error) {
            console.log(error);
        }
    });
    return result.result;
}

/**
 * Posts a new HyperCausal parameters build
 * @param {*} hyperCausalDto 
 * @returns 
 */
async function postNewHyperCausal(hyperCausalDto) {
    const result = await $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: '/api/hyper_causal/new',
        data: JSON.stringify(hyperCausalDto),
        error: function (error) {
            alert('There was a problem sending the request: ' + error);
            console.log(error);
        }
    });
    return result;
}

export { getNextTokenBranches, postNewHyperCausal };