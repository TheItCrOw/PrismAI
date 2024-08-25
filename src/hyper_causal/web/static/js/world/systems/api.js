
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

export { getNextTokenBranches };