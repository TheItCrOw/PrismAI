/**
 * Handle the switching of the different views
 */
$("body").on("click", "nav .nav-buttons button", function () {
    const targetView = $(this).data("id");
    switchToView(targetView);
})

function switchToView(targetView) {
    $("body .view").each(function () {
        if ($(this).data("id") === targetView) {
            $(this).show(75);
        } else {
            $(this).hide();
        }
    });
}

$(document).ready(function () {
    console.log("Reached PrismAI Gateway.")
})