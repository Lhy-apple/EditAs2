/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:34:40 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.NumberSerializer;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.io.File;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Month;
import java.util.ArrayList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberSerializer_ESTest extends NumberSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<BigDecimal> class0 = BigDecimal.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Class<Month> class1 = Month.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class1, (List<JavaType>) arrayList0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      numberSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, referenceType0);
      assertFalse(referenceType0.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = numberSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      Byte byte0 = new Byte((byte) (-123));
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", true);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      numberSerializer0.serialize((Number) byte0, jsonGenerator0, serializerProvider0);
      assertEquals(4, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      BigDecimal bigDecimal0 = new BigDecimal((-11L));
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("JSON", "JSON");
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF16_BE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator(file0, jsonEncoding0);
      numberSerializer0.serialize((Number) bigDecimal0, jsonGenerator0, (SerializerProvider) null);
      assertFalse(numberSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      BigInteger bigInteger0 = BigInteger.TEN;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        numberSerializer0.serialize((Number) bigInteger0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.NumberSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", true);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      numberSerializer0.serialize((Number) 55296, jsonGenerator0, serializerProvider0);
      assertEquals(5, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Long long0 = new Long((-391L));
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
  public void test07()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      Double double0 = new Double((-2068.20464668155));
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
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
  public void test08()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", true);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      Float float0 = new Float(0.0);
      numberSerializer0.serialize((Number) float0, jsonGenerator0, serializerProvider0);
      assertEquals(3, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", true);
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) mockFileOutputStream0, jsonEncoding0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      Short short0 = new Short((byte) (-123));
      numberSerializer0.serialize((Number) short0, jsonGenerator0, serializerProvider0);
      assertEquals(4, jsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
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
  public void test11()  throws Throwable  {
      Class<Double> class0 = Double.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProvider();
      JsonNode jsonNode0 = numberSerializer0.getSchema(serializerProvider0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      NumberSerializer numberSerializer0 = NumberSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<BigInteger> class0 = BigInteger.class;
      NumberSerializer numberSerializer0 = new NumberSerializer(class0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      numberSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(numberSerializer0.isUnwrappingSerializer());
  }
}