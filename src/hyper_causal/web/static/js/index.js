
/*
 * Insert the clicked popular llm name into the value field 
 */
$('body').on('click', '.llm-options a', function () {
    $('.llm-name-input').val($(this).html());
})