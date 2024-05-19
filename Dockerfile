FROM maven:3.8.4-openjdk-17-slim as build

WORKDIR /usr/src/app
COPY . .

RUN mvn -Dmaven.test.skip=true package

FROM openjdk:17-jdk-slim

WORKDIR /usr/src/app

COPY --from=build /usr/src/app/target/digitclassification-0.0.1-SNAPSHOT.jar .
COPY models /usr/src/app/models

EXPOSE 8080
CMD ["java","-jar","digitclassification-0.0.1-SNAPSHOT.jar"]