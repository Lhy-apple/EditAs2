/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:50:06 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.PropertySerializerMap;
import com.fasterxml.jackson.databind.ser.std.StdKeySerializers;
import java.io.IOException;
import java.io.StringWriter;
import java.util.Date;
import java.util.UUID;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.util.MockCalendar;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdKeySerializers_ESTest extends StdKeySerializers_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getDefault();
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic1 = (StdKeySerializers.Dynamic)stdKeySerializers_Dynamic0.readResolve();
      assertFalse(stdKeySerializers_Dynamic1.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StdKeySerializers.StringKeySerializer stdKeySerializers_StringKeySerializer0 = new StdKeySerializers.StringKeySerializer();
      assertFalse(stdKeySerializers_StringKeySerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.DEFAULT_STRING_SERIALIZER;
      Integer integer0 = new Integer(472);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, stdKeySerializers_Dynamic0, true);
      StringWriter stringWriter0 = new StringWriter();
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, stringWriter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        jsonSerializer0.serialize(integer0, writerBasedJsonGenerator0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Integer cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdKeySerializers$StringKeySerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, (Class<?>) null, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      PropertySerializerMap propertySerializerMap0 = PropertySerializerMap.emptyForRootValues();
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonSerializer<Object> jsonSerializer0 = stdKeySerializers_Dynamic0._findAndAddDynamic(propertySerializerMap0, class0, serializerProvider0);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Long> class0 = Long.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, true);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, serializerProvider0, true);
      MockFile mockFile0 = new MockFile("~* 5%vz 6P+jV/S");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, objectMapper0, mockPrintStream0);
      stdKeySerializers_Dynamic0.serialize(class0, uTF8JsonGenerator0, serializerProvider0);
      assertEquals(5, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Date> class0 = Date.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, false);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<MockCalendar> class0 = MockCalendar.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<UUID> class0 = UUID.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, false);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<ObjectMapper.DefaultTyping> class0 = ObjectMapper.DefaultTyping.class;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getStdKeySerializer((SerializationConfig) null, class0, true);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getFallbackKeySerializer((SerializationConfig) null, (Class<?>) null);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      JsonSerializer<Object> jsonSerializer0 = StdKeySerializers.getFallbackKeySerializer((SerializationConfig) null, class0);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      SerializationFeature serializationFeature0 = SerializationFeature.WRITE_DATES_AS_TIMESTAMPS;
      IOContext iOContext0 = new IOContext(bufferRecycler0, serializationFeature0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", false);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, mockFileOutputStream0);
      Class<AnnotationIntrospector.ReferenceProperty.Type> class0 = AnnotationIntrospector.ReferenceProperty.Type.class;
      StdKeySerializers.Default stdKeySerializers_Default0 = new StdKeySerializers.Default(1, class0);
      // Undeclared exception!
      try { 
        stdKeySerializers_Default0.serialize(objectMapper0, uTF8JsonGenerator0, serializerProvider0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // com.fasterxml.jackson.databind.ObjectMapper cannot be cast to java.util.Date
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdKeySerializers$Default", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      StdKeySerializers.Default stdKeySerializers_Default0 = new StdKeySerializers.Default(2, class0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate((JsonGenerator) null, true);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdKeySerializers_Default0.serialize(class0, jsonGeneratorDelegate0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.Class cannot be cast to java.util.Calendar
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdKeySerializers$Default", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      StdKeySerializers.Default stdKeySerializers_Default0 = new StdKeySerializers.Default(5, class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdKeySerializers_Default0.serialize((Object) null, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Object> class0 = Object.class;
      StdKeySerializers.Default stdKeySerializers_Default0 = new StdKeySerializers.Default(0, class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdKeySerializers_Default0.serialize(class0, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdKeySerializers$Default", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      Byte byte0 = new Byte((byte) (-58));
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      SerializationFeature serializationFeature0 = SerializationFeature.WRITE_ENUMS_USING_TO_STRING;
      IOContext iOContext0 = new IOContext(bufferRecycler0, byte0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper objectMapper1 = objectMapper0.enable(serializationFeature0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("~`c0Q!Jly", true);
      SerializerProvider serializerProvider0 = objectMapper1.getSerializerProviderInstance();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, objectMapper1, mockFileOutputStream0);
      stdKeySerializers_Dynamic0.serialize(serializationFeature0, uTF8JsonGenerator0, serializerProvider0);
      assertEquals(62, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StdKeySerializers.Dynamic stdKeySerializers_Dynamic0 = new StdKeySerializers.Dynamic();
      Byte byte0 = new Byte((byte) (-58));
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, byte0, true);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF16_BE;
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("~`c0Q!Jly", true);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2048, objectMapper0, mockFileOutputStream0);
      stdKeySerializers_Dynamic0.serialize(jsonEncoding0, uTF8JsonGenerator0, serializerProvider0);
      try { 
        stdKeySerializers_Dynamic0.serialize(jsonEncoding0, uTF8JsonGenerator0, serializerProvider0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not write a field name, expecting a value
         //
         verifyException("com.fasterxml.jackson.core.JsonGenerator", e);
      }
  }
}
