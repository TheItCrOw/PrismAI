
/*
 * Gets the next token branches from a given input
 * @param {} input 
 * @param {*} callback 
 */
async function getNextTokenBranches(input) {
    const result = await $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: '/api/tokens/next',
        data: JSON.stringify({ input: input }),
        error: function (error) {
            console.log(error);
        }
    });
    return result.result;
}

export { getNextTokenBranches };