import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import MapView, { PROVIDER_GOOGLE } from 'react-native-maps';
// import point from './point.json';
class HomeMapItem extends React.Component {

    render() {
        // const myPlace = {
        //         type: 'FeatureCollection',
        //         features: [
        //           {
        //             type: 'Feature',
        //             properties: {},
        //             geometry: {
        //               type: 'Point',
        //               coordinates: [64.165329, 48.844287],
        //             }
        //           }
        //         ]
        //       };
        return (
            <View>
                <MapView
                    provider={PROVIDER_GOOGLE}
                    showsUserLocation
                    initialRegion={{
                        latitude: 37.78825,
                        longitude: -122.4324,
                        latitudeDelta: 0.0922,
                        longitudeDelta: 0.0421
                    }}
                />
                <Text>hi</Text>
            </View>


        );

    }

}
const myStyle = StyleSheet.create({
    epmtyWarp: {
        alignItems: 'center',
        justifyContent: 'center'
    },
    epmtyItem: {
        textAlign: 'center',
        fontSize: 32
    },
})
export default HomeMapItem;