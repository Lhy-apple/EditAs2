/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:29:22 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import java.lang.reflect.Array;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Class<Object> class1 = Object.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      numberSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, simpleType0);
      assertFalse(numberSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BigInteger bigInteger0 = BigInteger.ZERO;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      Float float0 = new Float((-794.92F));
      IOContext iOContext0 = new IOContext(bufferRecycler0, float0, true);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      ObjectReader objectReader0 = objectMapper0.readerWithView(class0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 2);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 4028, objectReader0, byteArrayBuilder0);
      numberSerializer0.serialize((Number) bigInteger0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(3, uTF8JsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) null, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      BigDecimal bigDecimal0 = BigDecimal.TEN;
      JsonFactory jsonFactory0 = new JsonFactory();
      MockPrintStream mockPrintStream0 = new MockPrintStream("JSON");
      UTF8JsonGenerator uTF8JsonGenerator0 = (UTF8JsonGenerator)jsonFactory0.createGenerator((OutputStream) mockPrintStream0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      numberSerializer0.serialize((Number) bigDecimal0, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(57343, GeneratorBase.SURR2_LAST);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<BigInteger> class0 = BigInteger.class;
      Class<Long>[] classArray0 = (Class<Long>[]) Array.newInstance(Class.class, 0);
      JavaType javaType0 = typeFactory0.constructParametricType(class0, classArray0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, javaType0, true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(1);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, objectMapper0, byteArrayBuilder0, byteArrayBuilder0.NO_BYTES, 0, true);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) 3, (JsonGenerator) uTF8JsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("com.fasterxml.jackson.core.io.NumberOutput", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Long long0 = new Long(0L);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) long0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
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
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Double double0 = new Double((-440.2));
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
  public void test07()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Float float0 = Float.valueOf(0.0F);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) float0, (JsonGenerator) null, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Byte byte0 = new Byte((byte)0);
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) byte0, (JsonGenerator) null, (SerializerProvider) null);
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
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Short short0 = new Short((short)2719);
      JsonFactory jsonFactory0 = new JsonFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(pipedOutputStream0, (short)2719);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) bufferedOutputStream0, jsonEncoding0);
      numberSerializer0.serialize((Number) short0, jsonGenerator0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertEquals(4, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) null, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base((SerializerProvider) null);
      Class<String> class1 = String.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, simpleType0);
      assertFalse(numberSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.usesObjectId());
  }
}