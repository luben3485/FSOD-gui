$( document ).ready(function() {
    const sound = document.getElementsByTagName('audio')[0]

    var canvas = document.getElementById('cnt_query_img_canvas');
    var ctx = canvas.getContext('2d');
    var x = 0;
    var y = 0;
    var width = 500;
    var height = 500;
    var imageObj = new Image();

    //get canvas tag position
    var element = document.getElementById('cnt_query_img_canvas');
    var elemLeft = 0;
    var elemTop = 0;

    while ( element ) {
        elemLeft += element.offsetLeft - element.scrollLeft + element.clientLeft;
        elemTop += element.offsetTop - element.scrollLeft + element.clientTop;
        element = element.offsetParent;
    }
    var elements = [];


    // Add event listener for `click` events.
    canvas.addEventListener('click', function(event) {
        var x = event.pageX - elemLeft,
            y = event.pageY - elemTop;
            //console.log(x,y);

        // Collision detection between clicked offset and element.
        elements.forEach(function(element) {
            //console.log(element);
            if (y > element.top && y < element.top + element.height
                && x > element.left && x < element.left + element.width) {
                alert('clicked an element');
            }
        });

    }, false);



    imageObj.onload = function() {
        ctx.drawImage(imageObj, x, y, width, height);
    };
    imageObj.src = 'static/img/black.jpg';


    $( "#support_image" ).click(function() {
        $.ajax({
            url: '/support_image',
            method: "POST",
            data:{'support_im':$('#shot_img').attr('src')}
        }).done(function (res){
            $('#cameraModal').modal('hide')

            $('#cnt_support_text').empty();
            $('#pre_support_text').empty();
             if(res.cnt_shot==10){
                 $( "#support_image").attr('disabled','disabled');
             }
            $('#cnt_support_text').append('<h4>#Shot: ' + res.cnt_shot + '</h4>');
            $("#cnt_query_img").attr("src",res.cnt_result_im_path);
            $(new Image(100,100)).attr('src', res.cnt_support_im_paths[res.cnt_shot-1]).appendTo($('#cnt_support_img')).fadeIn();

            if(res.pre_shot != 0 && res.pre_shot<=9){
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

    $( "#add_a_shot" ).click(function() {
        $.ajax({
            url: '/add_a_shot',
            method: "POST",
        }).done(function (res){
            //previous method
            //$("#shot_img").attr("src","data:image/jpg;base64,"+res);

            //for refresh image
            $("#shot_img_1").attr("src","");
            $("#shot_img_2").attr("src","");

            //display support image
            $("#shot_img_1").attr("src",res.shot_img_1);
            $("#shot_img_2").attr("src",res.shot_img_2);
            $('#cameraModal').modal('show');

        });

    });

    $( "#query_image" ).click(function() {
        $.ajax({
            url: '/query_image',
            method: "GET",
        }).done(function (res) {
            $("#support_image").removeAttr('disabled');
            $('#cnt_support_text').empty();
            $('#pre_support_text').empty();
            $('#cnt_support_img').empty();
            $('#pre_support_img').empty();
            $("#cnt_query_img").attr("src","data:image/jpg;base64,"+res);
            $("#pre_query_img").attr("src","static/img/black.jpg");

        });
    });

    $( "#inference_model" ).click(function() {
        $.ajax({
            url: '/inference_model',
            method: "POST",
        }).done(function (res){


            //empty the text div
            $('#cnt_support_text').empty();
            $('#pre_support_text').empty();

            //current text & image
            $('#cnt_support_text').append('<h4>#Shot: ' + res.cnt_shot + '</h4>');
            $("#cnt_query_img").attr("src",res.cnt_result_im_path);
            $(new Image(100,100)).attr('src', res.cnt_support_im_paths[res.cnt_shot-1]).appendTo($('#cnt_support_img')).fadeIn();

            //previous text & image
            if(res.pre_shot != 0){
                $('#pre_support_text').append('<h4>#Shot: ' + res.pre_shot + '</h4>');
                $('#pre_query_img').attr("src",res.pre_result_im_path);
                $(new Image(100,100)).attr('src', res.pre_support_im_paths[res.pre_shot-1]).appendTo($('#pre_support_img')).fadeIn();
            }

        });
    });

    $( "#reset" ).click(function() {
        $('#cnt_support_text').empty();
        $('#pre_support_text').empty();
        $('#cnt_support_img').empty();
        $('#pre_support_img').empty();
        $("#cnt_query_img").attr("src","static/img/black.jpg");
        $("#pre_query_img").attr("src","static/img/black.jpg");
        $.ajax({
            url: '/reset',
            method: "POST",
        }).done(function (res){

        });

    });

    $( "#show" ).click(function() {


        $.ajax({
            url: '/show',
            method: "POST",
        }).done(function (res){
            $('#cameraModal').modal('show');
            black = 'static/img/black.jpg';
            var path = res.support_im_path
            var i;
            for (i = 0; i < path.length; i++) {
                (function(index){
                    setTimeout(function(){
                        $('#shot_img_1').css('opacity','0');
                        $('#shot_img_2').css('opacity','0');
                    },0);
                    setTimeout(function(){
                        sound.play()
                        $('#shot_img_1').css('opacity','1');
                        $('#shot_img_2').css('opacity','1');
                        $('#shot_img_2').attr('src',path[index]);
                        $('#shot_img_1').attr('src',black);
                        $('#shot_1').css('z-index','0');
                        $('#shot_2').css('z-index','1');
                    },1000*(index+1)-50);
                    setTimeout(function(){
                        $('#shot_1').css('z-index','2');

                    },1000*(index+1));
                    setTimeout(function(){
                         $('#shot_1').css('z-index','0');

                    },1000*(index+1)+50);

                })(i);
            }


        });


    });

    $( "#select_bbox_test" ).click(function() {

        $.ajax({
            url: '/select_bbox_test',
            method: "POST",
        }).done(function (res){
            console.log(res.bbox);
            console.log(res.query_path)
            //$("#cnt_query_img").attr("src",res.query_path);

            var imageObj = new Image();
            imageObj.onload = function() {
                ctx.drawImage(imageObj, x, y, width, height);
                var bbox = res.bbox
                bbox.forEach((el,index)=>{
                    ctx.beginPath();
                    ctx.lineWidth = "6";
                    ctx.strokeStyle = "red";
                    x = el[0];
                    y = el[1];
                    w = el[2] - el[0];
                    h = el[3] - el[1];
                    ctx.rect(x, y, w, h);
                    ctx.stroke();

                    // Add element.
                    elements.push({
                        width: w,
                        height: h,
                        top: x,
                        left: y
                    });

                });

            };
            imageObj.src = res.query_path;








        });

    });


});