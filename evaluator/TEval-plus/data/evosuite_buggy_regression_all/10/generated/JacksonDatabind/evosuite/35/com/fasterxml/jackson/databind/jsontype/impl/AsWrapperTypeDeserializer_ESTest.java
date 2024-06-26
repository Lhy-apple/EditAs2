/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:31:04 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.node.BinaryNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.IOException;
import java.io.PipedReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AsWrapperTypeDeserializer_ESTest extends AsWrapperTypeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.TYPE;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionLikeType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, minimalClassNameIdResolver0, "<UE;!b", true, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, minimalClassNameIdResolver0, true);
      PipedReader pipedReader0 = new PipedReader();
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromScalar(readerBasedJsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected token (null), expected START_OBJECT: need JSON Object to contain As.WRAPPER_OBJECT type information for class int
         //  at [Source: com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver@0000000023; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Object> class0 = Object.class;
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionLikeType0, typeFactory0);
      Class<BinaryNode> class1 = BinaryNode.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, minimalClassNameIdResolver0, "S,CFh1ES+", false, class1);
      JsonTypeInfo.As jsonTypeInfo_As0 = asWrapperTypeDeserializer0.getTypeInclusion();
      assertEquals(JsonTypeInfo.As.WRAPPER_OBJECT, jsonTypeInfo_As0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.TYPE;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionLikeType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, minimalClassNameIdResolver0, "]yh4~-", true, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectMapper0, true);
      PipedReader pipedReader0 = new PipedReader(2);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromArray(readerBasedJsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Integer> class0 = Integer.TYPE;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionLikeType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, minimalClassNameIdResolver0, "]yh4~-", true, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectMapper0, true);
      PipedReader pipedReader0 = new PipedReader(2);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, pipedReader0, objectMapper0, charsToNameCanonicalizer0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromAny(readerBasedJsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected token (null), expected START_OBJECT: need JSON Object to contain As.WRAPPER_OBJECT type information for class int
         //  at [Source: com.fasterxml.jackson.databind.ObjectMapper@0000000024; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectReader objectReader0 = objectMapper0.reader(jsonNodeFactory0);
      Class<Integer> class0 = Integer.TYPE;
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      CollectionLikeType collectionLikeType0 = typeFactory0.constructRawCollectionLikeType(class0);
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(collectionLikeType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, minimalClassNameIdResolver0, "]y~-", true, class0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectMapper0, true);
      PipedReader pipedReader0 = new PipedReader(3);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, pipedReader0, objectReader0, charsToNameCanonicalizer0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromObject(readerBasedJsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Class<String> class0 = String.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, javaType0, javaType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(mapLikeType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(mapLikeType0, minimalClassNameIdResolver0, "expected closing END_OBJECT after type information and deserialized value", true, class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, mapLikeType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 1213, minimalClassNameIdResolver0, propertyMetadata0);
      assertEquals(1213, creatorProperty0.getCreatorIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Float> class0 = Float.TYPE;
      JavaType javaType0 = objectMapper0.constructType(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(javaType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(javaType0, minimalClassNameIdResolver0, "JSON", false, class0);
      TypeDeserializer typeDeserializer0 = asWrapperTypeDeserializer0.forProperty((BeanProperty) null);
      assertSame(typeDeserializer0, asWrapperTypeDeserializer0);
  }
}
