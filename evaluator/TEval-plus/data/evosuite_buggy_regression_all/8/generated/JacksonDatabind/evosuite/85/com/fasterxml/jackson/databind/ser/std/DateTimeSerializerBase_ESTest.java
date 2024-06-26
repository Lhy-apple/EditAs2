/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:17:59 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.ser.std.CalendarSerializer;
import com.fasterxml.jackson.databind.ser.std.DateSerializer;
import com.fasterxml.jackson.databind.ser.std.SqlDateSerializer;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.lang.reflect.Type;
import java.text.DateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockDateFormat;
import org.evosuite.runtime.mock.java.util.MockCalendar;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DateTimeSerializerBase_ESTest extends DateTimeSerializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<SimpleModule> class0 = SimpleModule.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(resolvedRecursiveType0, resolvedRecursiveType0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(collectionLikeType0, (TypeIdResolver) null, "cQ-s,(BCR:XT&{Ni)a", true, collectionLikeType0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_REQUIRED_OR_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, resolvedRecursiveType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 1809, calendarSerializer0, propertyMetadata0);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonSerializer<?> jsonSerializer0 = calendarSerializer0.createContextual(serializerProvider0, creatorProperty0);
      assertSame(calendarSerializer0, jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<TextNode> class0 = TextNode.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, (TypeBindings) null);
      DateSerializer dateSerializer0 = new DateSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      // Undeclared exception!
      try { 
        dateSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, resolvedRecursiveType0);
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
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      JsonSerializer<?> jsonSerializer0 = calendarSerializer0.createContextual((SerializerProvider) null, (BeanProperty) null);
      assertSame(calendarSerializer0, jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      boolean boolean0 = dateSerializer0.isEmpty((Date) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar(2144101783, 2144101783, 2144101783, (-4172), 2144101783, (-483));
      boolean boolean0 = calendarSerializer0.isEmpty((Calendar) mockGregorianCalendar0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = CalendarSerializer.instance;
      Locale locale0 = Locale.US;
      Calendar calendar0 = MockCalendar.getInstance(locale0);
      boolean boolean0 = calendarSerializer0.isEmpty(calendar0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      boolean boolean0 = calendarSerializer0.isEmpty((SerializerProvider) null, (Calendar) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar((-2147483646), (-354), (-2147483646), (-354), 2143957402);
      boolean boolean0 = calendarSerializer0.isEmpty((SerializerProvider) null, (Calendar) mockGregorianCalendar0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar();
      CalendarSerializer calendarSerializer0 = CalendarSerializer.instance;
      boolean boolean0 = calendarSerializer0.isEmpty((SerializerProvider) null, (Calendar) mockGregorianCalendar0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonNode jsonNode0 = dateSerializer0.getSchema(serializerProvider0, (Type) null);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateSerializer dateSerializer0 = DateSerializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      Boolean boolean0 = Boolean.valueOf(true);
      DateFormat dateFormat0 = MockDateFormat.getDateTimeInstance();
      DateSerializer dateSerializer1 = dateSerializer0.withFormat(boolean0, dateFormat0);
      JsonNode jsonNode0 = dateSerializer1.getSchema(serializerProvider0, (Type) null);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      DateSerializer dateSerializer0 = DateSerializer.instance;
      Class<JsonTypeInfo.As> class0 = JsonTypeInfo.As.class;
      DateFormat dateFormat0 = MockDateFormat.getTimeInstance();
      DateSerializer dateSerializer1 = dateSerializer0.withFormat((Boolean) null, dateFormat0);
      JsonNode jsonNode0 = dateSerializer1.getSchema(serializerProvider0, (Type) class0);
      assertEquals(1, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SqlDateSerializer sqlDateSerializer0 = new SqlDateSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      sqlDateSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null);
      assertFalse(sqlDateSerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CalendarSerializer calendarSerializer0 = new CalendarSerializer();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      calendarSerializer0._acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, (JavaType) null, true);
      assertFalse(calendarSerializer0.usesObjectId());
  }
}
