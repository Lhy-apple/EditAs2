/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:07 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) null, (JsonGenerator) null, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Long long0 = new Long(1L);
      JsonFactory jsonFactory0 = new JsonFactory();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, jsonFactory0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(byteArrayOutputStream0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 0, objectMapper0, bufferedOutputStream0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializer0.serialize((Number) long0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(1, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFactory jsonFactory0 = new JsonFactory();
      MockFile mockFile0 = new MockFile("JSON");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF16_BE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(jsonGenerator0, true);
      BigDecimal bigDecimal0 = new BigDecimal((-2.147483648E9));
      numberSerializer0.serialize((Number) bigDecimal0, (JsonGenerator) jsonGeneratorDelegate0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(11, jsonGeneratorDelegate0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      byte[] byteArray0 = new byte[3];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) bigInteger0, (JsonGenerator) null, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFactory jsonFactory0 = new JsonFactory();
      MockFile mockFile0 = new MockFile("JSON");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(mockFile0);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      numberSerializer0.serialize((Number) 55296, jsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(5, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Double double0 = new Double(0.0);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) double0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFactory jsonFactory0 = new JsonFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) byteArrayOutputStream0, jsonEncoding0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(jsonGenerator0);
      Float float0 = new Float(0.0);
      numberSerializer0.serialize((Number) float0, (JsonGenerator) jsonGeneratorDelegate0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(3, jsonGeneratorDelegate0.getOutputBuffered());
      assertEquals(3, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, numberSerializer0, true);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 2, objectMapper0, byteArrayBuilder0, byteArray0, 0, true);
      numberSerializer0.serialize((Number) (byte)1, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertArrayEquals(new byte[] {(byte)49, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(1, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Short short0 = new Short((short)1);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) short0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Class<Float> class1 = Float.class;
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) null, (Type) class1);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base((SerializerProvider) null);
      JavaType javaType0 = TypeFactory.unknownType();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, javaType0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Float> class0 = Float.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.isUnwrappingSerializer());
  }
}