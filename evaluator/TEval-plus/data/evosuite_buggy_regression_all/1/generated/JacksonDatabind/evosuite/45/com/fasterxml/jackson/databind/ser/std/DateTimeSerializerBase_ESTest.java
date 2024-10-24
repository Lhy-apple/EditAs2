/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:37:18 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.DateSerializer;
import com.fasterxml.jackson.databind.ser.std.SqlDateSerializer;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Type;
import java.text.DateFormat;
import java.util.Date;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DateTimeSerializerBase_ESTest extends DateTimeSerializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      Class<Object> class0 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      sqlDateSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, resolvedRecursiveType0);
      assertFalse(resolvedRecursiveType0.isCollectionLikeType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        dateSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, javaType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null SerializerProvider passed for java.util.Date
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.DateTimeSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      JsonSerializer<?> jsonSerializer0 = dateSerializer0.createContextual((SerializerProvider) null, (BeanProperty) null);
      assertFalse(jsonSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      boolean boolean0 = dateSerializer0.isEmpty((Date) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      MockDate mockDate0 = new MockDate();
      boolean boolean0 = dateSerializer0.isEmpty((Date) mockDate0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      MockDate mockDate0 = new MockDate();
      mockDate0.setTime(0L);
      boolean boolean0 = dateSerializer0.isEmpty((Date) mockDate0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      boolean boolean0 = dateSerializer0.isEmpty((SerializerProvider) defaultSerializerProvider_Impl0, (Date) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      MockDate mockDate0 = new MockDate();
      boolean boolean0 = dateSerializer0.isEmpty((SerializerProvider) null, (Date) mockDate0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      MockDate mockDate0 = new MockDate();
      mockDate0.setTime(0);
      boolean boolean0 = dateSerializer0.isEmpty((SerializerProvider) null, (Date) mockDate0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Long> class0 = Long.TYPE;
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Boolean boolean0 = Boolean.valueOf(false);
      DateSerializer dateSerializer0 = new DateSerializer(boolean0, (DateFormat) null);
      JsonNode jsonNode0 = dateSerializer0.getSchema(serializerProvider0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateSerializer dateSerializer0 = new DateSerializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Long> class0 = Long.TYPE;
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonNode jsonNode0 = dateSerializer0.getSchema(serializerProvider0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      DateFormat dateFormat0 = MockDateFormat.getTimeInstance();
      DateSerializer dateSerializer0 = new DateSerializer((Boolean) null, dateFormat0);
      boolean boolean0 = dateSerializer0._asTimestamp(defaultSerializerProvider_Impl0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Boolean boolean0 = Boolean.TRUE;
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer(boolean0);
      sqlDateSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, (JavaType) null);
      assertFalse(sqlDateSerializer0.isUnwrappingSerializer());
  }
}
