import React from 'react';
import { View, Text } from 'react-native';

// function HomeItemDetailsItem(props) {
//   const { point } = props;

//   return (
//     <View>
//       <Text>景點：</Text>
//       <Text>{point.properties.name}</Text>
//     </View>
//   )
// }

const HomeItemDetailsItem = (props) => {
  const { id } = props;

  return (
    <View>
      
      <Text>{id}</Text>
    </View>
  )
}

export default HomeItemDetailsItem;