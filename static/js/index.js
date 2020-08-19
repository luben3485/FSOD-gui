$( document ).ready(function() {
    $( "#support_image" ).click(function() {
        $.ajax({
            url: '/support_image',
            method: "GET",
        }).done(function (res){
            //$("body").html(res)
            $('#cnt_support_text').empty();
            $('#pre_support_text').empty();
            if(res.cnt_shot==5){
                $( "#support_image").attr('disabled','disabled');
            }
            $('#cnt_support_text').append('<h4>#Shot: ' + res.cnt_shot + '</h4>');
            $("#cnt_query_img").attr("src",res.cnt_result_im_path);
            $(new Image(100,100)).attr('src', res.cnt_support_im_paths[res.cnt_shot-1]).appendTo($('#cnt_support_img')).fadeIn();

            if(res.pre_shot != 0 && res.pre_shot<=4){
                $('#pre_support_text').append('<h4>#Shot: ' + res.pre_shot + '</h4>');
                $("#pre_query_img").attr("src",res.pre_result_im_path);
                $(new Image(100,100)).attr('src', res.pre_support_im_paths[res.pre_shot-1]).appendTo($('#pre_support_img')).fadeIn();
            }


//            var i;
//            for (i = 0; i < res.cnt_support_im_paths.length; i++) {
//                $(new Image(100,100)).attr('src', res.cnt_support_im_paths[i]).appendTo($('#cnt_support')).fadeIn();
//            }
//            for (i = 0; i < res.pre_support_im_paths.length; i++) {
//                $(new Image(100,100)).attr('src', res.pre_support_im_paths[i]).appendTo($('#pre_support')).fadeIn();
//            }




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
            //$("body").html(res)
            //$("#previous_img").attr("src","data:image/jpg;base64,"+data);
            //$("#cnt_query_img").attr("src","data:image/jpg;base64,"+data);
            $("#support_image").removeAttr('disabled');
            $('#cnt_support_text').empty();
            $('#pre_support_text').empty();
            $('#cnt_support_img').empty();
            $('#pre_support_img').empty();
            $("#cnt_query_img").attr("src",res.query_path);
            $("#pre_query_img").attr("src","static/img/black.jpg");


        });
    });




});