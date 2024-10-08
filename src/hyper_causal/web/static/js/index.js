import { postNewHyperCausal } from './world/systems/api.js';

/*
 * Insert the clicked popular llm name into the value field 
 */
$('body').on('click', '.llm-options a', function () {
    $('.llm-name-input').val($(this).html());
})

/*
 * Open HyperCausal instance 
 */
$('body').on('click', '.instances-list .instance-item .open-hyper-causal-btn', function () {
    const id = $(this).data('id');
    window.open(`/hyper_causal?id=${id}`, '_blank');
})

/**
 * Switch between different navbar views
 */
$('body').on('click', '.navbar button', function () {
    const target = $(this).data('id');
    $('.navbar button').each(function () { $(this).removeClass('selected-navbar-btn') })
    $(this).addClass('selected-navbar-btn');
    $('.views .params-container').each(function () {
        $(this).hide();
    });
    $(`.views .params-container[data-id="${target}"]`).fadeIn(50);
})

/**
 * Generates a new hypercausal instance and opens a new tab for it.
 */
$('body').on('click', '.generate-hypercausal-btn', async function () {
    // Indicate loading
    $(this).find('i').removeClass('fa-project-diagram');
    $(this).find('i').addClass('fa-spinner rotate');
    try {
        // First, gather the parameters
        const $container = $('.params-container');
        const obj = {
            llm: $container.find('.llm-name-input').val(),
            input: $container.find('.prompt-input').val(),
            k: $container.find('.k-input').val(),
            temp: $container.find('.temp-input').val(),
            maxTokens: $container.find('.max-tokens-input').val(),
            treeStyle: $container.find('.tree-style-select').val(),
            decodingStrategy: $container.find('.decoding-strategy-select').val(),
            p: $container.find('.p-input').val(),
            beamWidth: $container.find('.beam-width-input').val()
        };

        // Post a new hypercausal instance to our backend, so we can reference it later.
        const result = await postNewHyperCausal(obj);

        if (result.status !== 200) {
            alert("Couldn't generate the HyperCausal, server sent: " + result.message);
            return;
        }

        // Else, we open a new tab with the new HyperCausal id
        //window.open(`/hyper_causal?id=${result.id}`, '_blank');
        // After we've stored the new hypercausal, we reload the page
        location.reload();
    } catch (ex) {
        alert('Unknown error occured client sided, has nothing to do with backend:\n ' + ex);
    } finally {
        // Remove loading
        $(this).find('i').addClass('fa-project-diagram');
        $(this).find('i').removeClass('fa-spinner rotate');
    }

})