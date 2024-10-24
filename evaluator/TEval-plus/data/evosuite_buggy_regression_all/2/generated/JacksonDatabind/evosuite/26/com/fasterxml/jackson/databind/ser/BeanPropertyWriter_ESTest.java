/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:25 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.io.SerializedString;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonObjectFormatVisitor;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.PropertySerializerMap;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.NameTransformer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.time.format.TextStyle;
import java.util.ArrayDeque;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanPropertyWriter_ESTest extends BeanPropertyWriter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Field> class0 = Field.class;
      ObjectWriter objectWriter0 = objectMapper0.writerFor(class0);
      assertTrue(objectWriter0.hasPrefetchedSerializer());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      PropertyName propertyName0 = PropertyName.construct(" |2>k7:{9=YwT`i", " |2>k7:{9=YwT`i");
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector1, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER, (TypeSerializer) null, (JavaType) null, false, annotationMap0);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertFalse(objectNode0.isLong());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.assignTypeSerializer((TypeSerializer) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      boolean boolean0 = beanPropertyWriter0.isVirtual();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      boolean boolean0 = beanPropertyWriter0.isUnwrapping();
      assertFalse(boolean0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getWrapperName();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      // Undeclared exception!
      try { 
        beanPropertyWriter0.getFullName();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getSerializedName();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getMetadata();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      BeanPropertyWriter beanPropertyWriter1 = beanPropertyWriter0.unwrappingWriter(nameTransformer0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      NameTransformer nameTransformer0 = NameTransformer.simpleTransformer("VRhb)|IB=@$s1n+}Rr", (String) null);
      NameTransformer nameTransformer1 = NameTransformer.chainedTransformer(nameTransformer0, nameTransformer0);
      BeanPropertyWriter beanPropertyWriter2 = beanPropertyWriter1.rename(nameTransformer1);
      assertFalse(beanPropertyWriter2.willSuppressNulls());
      assertEquals("VRhb)|IB=@$s1n+}RrVRhb)|IB=@$s1n+}Rr", beanPropertyWriter2.getName());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      beanPropertyWriter0._internalSettings = hashMap0;
      PropertyName propertyName0 = PropertyName.construct(";hc&s");
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      beanPropertyWriter0._internalSettings = hashMap0;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, serializedString0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, serializedString0);
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      BeanPropertyWriter beanPropertyWriter2 = beanPropertyWriter1.rename(nameTransformer0);
      assertSame(beanPropertyWriter2, beanPropertyWriter1);
      assertFalse(beanPropertyWriter2.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName((String) null);
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<TextStyle> jsonSerializer0 = (JsonSerializer<TextStyle>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, (TypeSerializer) null, (JavaType) null, true, (Object) null);
      // Undeclared exception!
      try { 
        beanPropertyWriter0.assignSerializer((JsonSerializer<Object>) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Can not override serializer
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertTrue(beanPropertyWriter0.hasSerializer());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      beanPropertyWriter0.assignNullSerializer(beanSerializer0);
      beanPropertyWriter0.assignNullSerializer(beanSerializer0);
      assertTrue(beanPropertyWriter0.hasNullSerializer());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      BeanSerializer beanSerializer1 = new BeanSerializer(beanSerializer0);
      beanPropertyWriter0.assignNullSerializer(beanSerializer0);
      // Undeclared exception!
      try { 
        beanPropertyWriter0.assignNullSerializer(beanSerializer1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Can not override null serializer
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      beanPropertyWriter0.readResolve();
      assertTrue(beanPropertyWriter0.hasSerializer());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<Annotation> class0 = Annotation.class;
      beanPropertyWriter0.findAnnotation(class0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<TextStyle> jsonSerializer0 = (JsonSerializer<TextStyle>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, (TypeSerializer) null, (JavaType) null, false, (Object) null);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = beanPropertyWriter0.getContextAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JsonFormat.Value jsonFormat_Value0 = beanPropertyWriter0.findFormatOverrides(annotationIntrospector0);
      assertNull(jsonFormat_Value0);
      
      JsonFormat.Value jsonFormat_Value1 = beanPropertyWriter0.findFormatOverrides(annotationIntrospector0);
      assertNull(jsonFormat_Value1);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JsonFormat.Value jsonFormat_Value0 = beanPropertyWriter0.findFormatOverrides((AnnotationIntrospector) null);
      assertNull(jsonFormat_Value0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0);
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      beanPropertyWriter1._internalSettings = hashMap0;
      PropertySerializerMap propertySerializerMap0 = PropertySerializerMap.emptyForRootValues();
      beanPropertyWriter1.getInternalSetting(propertySerializerMap0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getInternalSetting((Object) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0);
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      beanPropertyWriter1._internalSettings = hashMap0;
      beanPropertyWriter1.setInternalSetting(beanPropertyWriter0, serializedString0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      beanPropertyWriter0.setInternalSetting("Qbj6adv", dOMSerializer0);
      Object object0 = beanPropertyWriter0.removeInternalSetting((Object) null);
      assertNull(object0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.removeInternalSetting((Object) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      HashMap<Object, Object> hashMap0 = new HashMap<Object, Object>();
      beanPropertyWriter0._internalSettings = hashMap0;
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      Object object0 = beanPropertyWriter0.removeInternalSetting(dOMSerializer0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      boolean boolean0 = beanPropertyWriter0.hasSerializer();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      boolean boolean0 = beanPropertyWriter0.hasNullSerializer();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      PropertyName propertyName1 = PropertyName.construct("VB3F.");
      boolean boolean0 = beanPropertyWriter1.wouldConflictWithName(propertyName1);
      assertFalse(boolean0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      boolean boolean0 = beanPropertyWriter1.wouldConflictWithName(propertyName0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("Z)F", "Z)F");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      boolean boolean0 = beanPropertyWriter1.wouldConflictWithName(propertyName0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector0, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      Class<ArrayDeque> class0 = ArrayDeque.class;
      Class<Object> class1 = Object.class;
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      CollectionType collectionType0 = CollectionType.construct(class1, simpleType0);
      JsonSerializer<Field> jsonSerializer0 = (JsonSerializer<Field>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, collectionType0, jsonSerializer0, (TypeSerializer) null, simpleType0, true, class0);
      Class<?> class2 = beanPropertyWriter0.getRawSerializationType();
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getRawSerializationType();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      // Undeclared exception!
      try { 
        beanPropertyWriter0.getPropertyType();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      try { 
        beanPropertyWriter0.serializeAsField(defaultSerializerProvider_Impl0, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, beanPropertyWriter0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(3);
      byte[] byteArray0 = new byte[1];
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 862, objectMapper0, byteArrayOutputStream0, byteArray0, 0, false);
      beanPropertyWriter0.serializeAsOmittedField(defaultSerializerProvider_Impl0, uTF8JsonGenerator0, defaultSerializerProvider_Impl0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      try { 
        beanPropertyWriter0.serializeAsElement((Object) null, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      try { 
        beanPropertyWriter0.serializeAsPlaceholder(defaultSerializerProvider_Impl0, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      Object object0 = new Object();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, object0, false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      MockFile mockFile0 = new MockFile("", "E2xIgJ+(fC");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 406, objectMapper0, mockPrintStream0);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(uTF8JsonGenerator0, true);
      try { 
        beanPropertyWriter0.serializeAsPlaceholder(object0, jsonGeneratorDelegate0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Null key for a Map not allowed in JSON (use a converting NullKeySerializer?)
         //
         verifyException("com.fasterxml.jackson.databind.ser.impl.FailingSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.depositSchemaProperty((JsonObjectFormatVisitor) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector0, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER, (TypeSerializer) null, (JavaType) null, true, "b");
      JsonObjectFormatVisitor.Base jsonObjectFormatVisitor_Base0 = new JsonObjectFormatVisitor.Base();
      beanPropertyWriter0.depositSchemaProperty((JsonObjectFormatVisitor) jsonObjectFormatVisitor_Base0);
      assertEquals("", beanPropertyWriter0.getName());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      // Undeclared exception!
      try { 
        beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<TextStyle> jsonSerializer0 = (JsonSerializer<TextStyle>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonSerializer0).toString();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, (TypeSerializer) null, (JavaType) null, false, (Object) null);
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertySerializerMap propertySerializerMap0 = PropertySerializerMap.emptyForRootValues();
      Class<Method> class0 = Method.class;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanPropertyWriter0._findAndAddDynamic(propertySerializerMap0, class0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<Annotation> class0 = Annotation.class;
      Class<Object> class1 = Object.class;
      Class<AnnotatedMethod> class2 = AnnotatedMethod.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapLikeType mapLikeType0 = MapLikeType.construct(class2, javaType0, javaType0);
      MapType mapType0 = MapType.construct(class1, mapLikeType0, javaType0);
      beanPropertyWriter0.setNonTrivialBaseType(mapType0);
      // Undeclared exception!
      try { 
        beanPropertyWriter0._findAndAddDynamic((PropertySerializerMap) null, class0, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      SerializedString serializedString0 = DefaultPrettyPrinter.DEFAULT_ROOT_VALUE_SEPARATOR;
      try { 
        beanPropertyWriter0.get(serializedString0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      PropertyName propertyName0 = PropertyName.construct(" |2>k7:{9=YwT`i", " |2>k7:{9=YwT`i");
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, annotationIntrospector1, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER, (TypeSerializer) null, (JavaType) null, false, annotationMap0);
      String string0 = beanPropertyWriter0.toString();
      assertEquals("property ' |2>k7:{9=YwT`i' (virtual, static serializer of type com.fasterxml.jackson.databind.ser.impl.FailingSerializer)", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      SerializedString serializedString0 = new SerializedString("(8Ju~<Yy[ ,+kC.aT");
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, serializedString0);
      String string0 = beanPropertyWriter1.toString();
      assertFalse(beanPropertyWriter1.willSuppressNulls());
      assertEquals("property '(8Ju~<Yy[ ,+kC.aT' (virtual, no static serializer)", string0);
  }
}
