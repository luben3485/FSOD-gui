$( document ).ready(function() {
    $( "#support_image" ).click(function() {
        $.ajax({
            url: '/support_image',
            method: "GET",
        }).done(function (res){
            $("body").html(res)
            //currnet_n_shot = data.n_shot;
            //$("#current_img").attr("src","data:image/jpg;base64,"+data.im_b64);
            //$('#previous_support').append('<h4>#Shot: ' + current_n_shot + '</h4>');
            //_filename = 'static/img/horse02.jpg';
            //$(new Image(100,100)).attr('src', '' + _filename).appendTo($('#previous_support')).fadeIn();
        });
    });

    $( "#query_image" ).click(function() {
        $.ajax({
            url: '/query_image',
            method: "GET",
        }).done(function (res) {
            $("body").html(res)
            //$("#previous_img").attr("src","data:image/jpg;base64,"+data);
            //$("#current_img").attr("src","data:image/jpg;base64,"+data);
        });
    });




});