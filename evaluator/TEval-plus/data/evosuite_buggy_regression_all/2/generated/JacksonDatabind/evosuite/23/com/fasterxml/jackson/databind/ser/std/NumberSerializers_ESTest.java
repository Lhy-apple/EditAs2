/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:03 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializers;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Type;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializers_ESTest extends NumberSerializers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer();
      Class<Float> class0 = Float.TYPE;
      JsonNode jsonNode0 = numberSerializers_IntegerSerializer0.getSchema((SerializerProvider) null, (Type) class0, false);
      assertFalse(jsonNode0.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberSerializers numberSerializers0 = new NumberSerializers();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer();
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_IntegerSerializer0.serializeWithType(dOMSerializer0, (JsonGenerator) null, defaultSerializerProvider_Impl0, (TypeSerializer) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.ext.DOMSerializer cannot be cast to java.lang.Integer
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntegerSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberSerializers.FloatSerializer numberSerializers_FloatSerializer0 = new NumberSerializers.FloatSerializer();
      Byte byte0 = new Byte((byte)72);
      // Undeclared exception!
      try { 
        numberSerializers_FloatSerializer0.serialize(byte0, (JsonGenerator) null, (SerializerProvider) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Byte cannot be cast to java.lang.Float
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$FloatSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer();
      NumberSerializers.LongSerializer numberSerializers_LongSerializer0 = new NumberSerializers.LongSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_LongSerializer0.serialize(numberSerializers_IntegerSerializer0, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntegerSerializer cannot be cast to java.lang.Long
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$LongSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NumberSerializers.IntLikeSerializer numberSerializers_IntLikeSerializer0 = NumberSerializers.IntLikeSerializer.instance;
      BigInteger bigInteger0 = BigInteger.ZERO;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "", false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationFeature deserializationFeature0 = DeserializationFeature.FAIL_ON_UNRESOLVED_OBJECT_IDS;
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0);
      MockFile mockFile0 = new MockFile("", "8<Y");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      byte[] byteArray0 = new byte[0];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectReader0, mockPrintStream0, byteArray0, 1, false);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_IntLikeSerializer0.serialize((Number) bigInteger0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("com.fasterxml.jackson.core.io.NumberOutput", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NumberSerializers.IntLikeSerializer numberSerializers_IntLikeSerializer0 = new NumberSerializers.IntLikeSerializer();
      assertFalse(numberSerializers_IntLikeSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NumberSerializers.DoubleSerializer numberSerializers_DoubleSerializer0 = new NumberSerializers.DoubleSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, jsonFormatVisitorWrapper_Base0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(1);
      DataOutputStream dataOutputStream0 = new DataOutputStream(byteArrayBuilder0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, (ObjectCodec) null, dataOutputStream0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_DoubleSerializer0.serializeWithType((Object) null, uTF8JsonGenerator0, defaultSerializerProvider_Impl0, (TypeSerializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$DoubleSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NumberSerializers.ShortSerializer numberSerializers_ShortSerializer0 = new NumberSerializers.ShortSerializer();
      Short short0 = new Short((short) (-1));
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, numberSerializers_ShortSerializer0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      byte[] byteArray0 = new byte[3];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, mockPrintStream0, byteArray0, 2, false);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0, false);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializers_ShortSerializer0.serialize(short0, (JsonGenerator) jsonGeneratorDelegate0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertFalse(jsonGeneratorDelegate0.canWriteTypeId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NumberSerializers.DoubleSerializer numberSerializers_DoubleSerializer0 = new NumberSerializers.DoubleSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializers_DoubleSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializers_DoubleSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializers_IntegerSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializers_IntegerSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer();
      JsonSerializer<?> jsonSerializer0 = numberSerializers_IntegerSerializer0.createContextual((SerializerProvider) null, (BeanProperty) null);
      assertSame(numberSerializers_IntegerSerializer0, jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      NumberSerializers.ShortSerializer numberSerializers_ShortSerializer0 = new NumberSerializers.ShortSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std((PropertyName) null, javaType0, (PropertyName) null, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      JsonSerializer<?> jsonSerializer0 = numberSerializers_ShortSerializer0.createContextual(defaultSerializerProvider_Impl0, beanProperty_Std0);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      NumberSerializers.ShortSerializer numberSerializers_ShortSerializer0 = NumberSerializers.ShortSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      PropertyName propertyName0 = PropertyName.construct("g@A\"t<%E!yy2@", "g@A\"t<%E!yy2@");
      Class<Double> class0 = Double.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(simpleType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, classNameIdResolver0, "g@A\"t<%E!yy2@", false, class0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, class0, (AnnotationMap) null, 0);
      Integer integer0 = new Integer(0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.construct(false, "W;$({H;7DUI[", integer0, "W;$({H;7DUI[");
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, annotatedParameter0, 0, "g@A\"t<%E!yy2@", propertyMetadata0);
      // Undeclared exception!
      try { 
        numberSerializers_ShortSerializer0.createContextual(defaultSerializerProvider_Impl0, creatorProperty0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }
}