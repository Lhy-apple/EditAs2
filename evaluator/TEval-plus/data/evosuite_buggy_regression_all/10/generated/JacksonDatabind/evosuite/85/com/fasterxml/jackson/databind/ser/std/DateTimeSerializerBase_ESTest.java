/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:40:01 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.CalendarSerializer;
import com.fasterxml.jackson.databind.ser.std.DateSerializer;
import com.fasterxml.jackson.databind.ser.std.SqlDateSerializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.sql.Date;
import java.text.DateFormat;
import java.util.Calendar;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DateTimeSerializerBase_ESTest extends DateTimeSerializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JavaType javaType0 = TypeFactory.unknownType();
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, javaType0, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonSerializer<?> jsonSerializer0 = sqlDateSerializer0.createContextual(serializerProvider0, beanProperty_Std0);
      assertSame(sqlDateSerializer0, jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DateSerializer dateSerializer0 = new DateSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base((SerializerProvider) null);
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      // Undeclared exception!
      try { 
        dateSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, referenceType0);
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
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonSerializer<?> jsonSerializer0 = sqlDateSerializer0.createContextual(defaultSerializerProvider_Impl0, (BeanProperty) null);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      boolean boolean0 = sqlDateSerializer0.isEmpty((Date) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      Date date0 = new Date(1019L);
      boolean boolean0 = sqlDateSerializer0.isEmpty(date0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      Date date0 = new Date(0L);
      boolean boolean0 = sqlDateSerializer0.isEmpty(date0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      boolean boolean0 = sqlDateSerializer0.isEmpty((SerializerProvider) null, (Date) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      Date date0 = new Date(1L);
      boolean boolean0 = sqlDateSerializer0.isEmpty((SerializerProvider) null, date0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      Date date0 = new Date(0L);
      boolean boolean0 = sqlDateSerializer0.isEmpty((SerializerProvider) null, date0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf("@OVw'1s%i");
      DateSerializer dateSerializer0 = new DateSerializer(boolean0, (DateFormat) null);
      Class<Integer> class0 = Integer.TYPE;
      JsonNode jsonNode0 = dateSerializer0.getSchema((SerializerProvider) null, (Type) class0, true);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateSerializer dateSerializer0 = new DateSerializer();
      Class<Integer> class0 = Integer.TYPE;
      Boolean boolean0 = Boolean.TRUE;
      DateFormat dateFormat0 = MockDateFormat.getDateTimeInstance();
      DateSerializer dateSerializer1 = dateSerializer0.withFormat(boolean0, dateFormat0);
      JsonNode jsonNode0 = dateSerializer1.getSchema((SerializerProvider) null, (Type) class0, true);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      CalendarSerializer calendarSerializer0 = CalendarSerializer.instance;
      DateFormat dateFormat0 = MockDateFormat.getInstance();
      CalendarSerializer calendarSerializer1 = calendarSerializer0.withFormat((Boolean) null, dateFormat0);
      TimeZone timeZone0 = serializerProvider0.getTimeZone();
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar(timeZone0);
      // Undeclared exception!
      try { 
        calendarSerializer1.serialize((Calendar) mockGregorianCalendar0, (JsonGenerator) null, serializerProvider0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.CalendarSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DateSerializer dateSerializer0 = new DateSerializer();
      MockDate mockDate0 = new MockDate();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        dateSerializer0.serialize((java.util.Date) mockDate0, (JsonGenerator) null, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      JavaType javaType0 = TypeFactory.unknownType();
      sqlDateSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, javaType0);
      assertFalse(javaType0.isAbstract());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<InputStream> class0 = InputStream.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      sqlDateSerializer0._acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, mapLikeType0, true);
      assertFalse(mapLikeType0.useStaticType());
  }
}
