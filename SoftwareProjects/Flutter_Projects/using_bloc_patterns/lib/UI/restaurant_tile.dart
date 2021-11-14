import 'package:flutter/material.dart';

import 'package:using_bloc_patterns/DataLayer/restaurant.dart';
import 'package:using_bloc_patterns/UI/image_container.dart';
import 'package:using_bloc_patterns/UI/restaurant_details_screen.dart';

class RestaurantTile extends StatelessWidget {
  final Restaurant restaurant;

  const RestaurantTile({
    Key key,
    @required this.restaurant
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: ImageContainer(
        width: 50,
        height: 50,
        url: restaurant.thumbUrl
      ),
      title: Text(
        restaurant.name
      ),
      trailing: Icon(
        Icons.keyboard_arrow_right
      ),
      onTap: () {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => RestaurantDetailsScreen(
              restaurant: restaurant,
            )
          )
        );
      },
    );
  }
}