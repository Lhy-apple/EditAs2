/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:46:56 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonFactoryBuilder;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.filter.FilteringGeneratorDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.ext.CoreXMLSerializers;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.ext.NioPathSerializer;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializers;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.DataOutput;
import java.io.FilterOutputStream;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import java.lang.reflect.Type;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializers_ESTest extends NumberSerializers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NumberSerializers.FloatSerializer numberSerializers_FloatSerializer0 = new NumberSerializers.FloatSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializers_FloatSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) null);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberSerializers numberSerializers0 = new NumberSerializers();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<CoreXMLSerializers.XMLGregorianCalendarSerializer> class0 = CoreXMLSerializers.XMLGregorianCalendarSerializer.class;
      NumberSerializers.IntegerSerializer numberSerializers_IntegerSerializer0 = new NumberSerializers.IntegerSerializer(class0);
      Double double0 = new Double(0.0);
      JsonFactoryBuilder jsonFactoryBuilder0 = new JsonFactoryBuilder();
      JsonFactory jsonFactory0 = new JsonFactory(jsonFactoryBuilder0);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((DataOutput) null);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringGeneratorDelegate filteringGeneratorDelegate0 = new FilteringGeneratorDelegate(jsonGenerator0, tokenFilter0, false, false);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_IntegerSerializer0.serializeWithType(double0, filteringGeneratorDelegate0, defaultSerializerProvider_Impl0, (TypeSerializer) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Double cannot be cast to java.lang.Integer
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntegerSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberSerializers.FloatSerializer numberSerializers_FloatSerializer0 = new NumberSerializers.FloatSerializer();
      Object object0 = new Object();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, numberSerializers_FloatSerializer0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 3);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, byteArrayBuilder0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0, true);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_FloatSerializer0.serialize(object0, jsonGeneratorDelegate0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Object cannot be cast to java.lang.Float
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$FloatSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<BeanSerializer> class0 = BeanSerializer.class;
      NumberSerializers.LongSerializer numberSerializers_LongSerializer0 = new NumberSerializers.LongSerializer(class0);
      // Undeclared exception!
      try { 
        numberSerializers_LongSerializer0.serialize(class0, (JsonGenerator) null, (SerializerProvider) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Class cannot be cast to java.lang.Long
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$LongSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NumberSerializers.IntLikeSerializer numberSerializers_IntLikeSerializer0 = NumberSerializers.IntLikeSerializer.instance;
      NioPathSerializer nioPathSerializer0 = new NioPathSerializer();
      JsonFactoryBuilder jsonFactoryBuilder0 = new JsonFactoryBuilder();
      JsonFactory jsonFactory0 = new JsonFactory(jsonFactoryBuilder0);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      FilterOutputStream filterOutputStream0 = new FilterOutputStream(pipedOutputStream0);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF16_LE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) filterOutputStream0, jsonEncoding0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(jsonGenerator0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_IntLikeSerializer0.serialize(nioPathSerializer0, jsonGeneratorDelegate0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.ext.NioPathSerializer cannot be cast to java.lang.Number
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$IntLikeSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NumberSerializers.IntLikeSerializer numberSerializers_IntLikeSerializer0 = new NumberSerializers.IntLikeSerializer();
      assertFalse(numberSerializers_IntLikeSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializers.DoubleSerializer numberSerializers_DoubleSerializer0 = new NumberSerializers.DoubleSerializer(class0);
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      Float float0 = new Float((float) 0);
      IOContext iOContext0 = new IOContext(bufferRecycler0, float0, false);
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      MockFile mockFile0 = new MockFile("JSON");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      byte[] byteArray0 = new byte[6];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectMapper0, mockPrintStream0, byteArray0, (byte)0, false);
      // Undeclared exception!
      try { 
        numberSerializers_DoubleSerializer0.serializeWithType(dOMSerializer0, uTF8JsonGenerator0, defaultSerializerProvider_Impl0, (TypeSerializer) null);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.ext.DOMSerializer cannot be cast to java.lang.Double
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$DoubleSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NumberSerializers.ShortSerializer numberSerializers_ShortSerializer0 = new NumberSerializers.ShortSerializer();
      Float float0 = new Float((-1777.824F));
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, float0, true);
      JsonFactoryBuilder jsonFactoryBuilder0 = new JsonFactoryBuilder();
      JsonFactory jsonFactory0 = new JsonFactory(jsonFactoryBuilder0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, (OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializers_ShortSerializer0.serialize(float0, uTF8JsonGenerator0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Float cannot be cast to java.lang.Short
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializers$ShortSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NumberSerializers.FloatSerializer numberSerializers_FloatSerializer0 = new NumberSerializers.FloatSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializers_FloatSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializers_FloatSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      NumberSerializers.ShortSerializer numberSerializers_ShortSerializer0 = new NumberSerializers.ShortSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<DOMSerializer> class0 = DOMSerializer.class;
      ArrayType arrayType0 = typeFactory0.constructArrayType(class0);
      numberSerializers_ShortSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, arrayType0);
      assertTrue(arrayType0.isArrayType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NumberSerializers.FloatSerializer numberSerializers_FloatSerializer0 = new NumberSerializers.FloatSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JsonSerializer<?> jsonSerializer0 = numberSerializers_FloatSerializer0.createContextual(defaultSerializerProvider_Impl0, beanProperty_Bogus0);
      assertSame(numberSerializers_FloatSerializer0, jsonSerializer0);
  }
}