/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:35:21 GMT 2023
 */

package com.fasterxml.jackson.databind.ser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ext.CoreXMLSerializers;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonObjectFormatVisitor;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.impl.PropertySerializerMap;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.NameTransformer;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import java.io.Writer;
import java.lang.annotation.Annotation;
import java.math.RoundingMode;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanPropertyWriter_ESTest extends BeanPropertyWriter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<RoundingMode> class0 = RoundingMode.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      // Undeclared exception!
      try { 
        objectMapper0.convertValue((Object) objectMapper0, class0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.BeanSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Boolean boolean0 = new Boolean("Z}|l^:RA}|l{:R");
      PropertyName propertyName0 = PropertyName.construct("Z}|l^:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, (boolean) boolean0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<String> jsonSerializer0 = (JsonSerializer<String>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonSerializer0).toString();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "Z}|l^:RA}|l{:R");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, asExternalTypeSerializer0, (JavaType) null, (boolean) boolean0, propertyName0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals((boolean) boolean0);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertFalse(objectNode0.isFloatingPointNumber());
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
      Class<RoundingMode> class0 = RoundingMode.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.ALL;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.NON_PRIVATE;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      // Undeclared exception!
      try { 
        objectMapper0.convertValue((Object) objectMapper1, class0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getSerializedName();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getMetadata();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      BeanPropertyWriter beanPropertyWriter1 = beanPropertyWriter0.unwrappingWriter(nameTransformer0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      boolean boolean0 = beanPropertyWriter1.wouldConflictWithName(propertyName0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      JsonInclude.Include jsonInclude_Include0 = (JsonInclude.Include)BeanPropertyWriter.MARKER_FOR_EMPTY;
      beanPropertyWriter0.setInternalSetting(propertyName0, jsonInclude_Include0);
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JsonInclude.Include jsonInclude_Include0 = (JsonInclude.Include)BeanPropertyWriter.MARKER_FOR_EMPTY;
      beanPropertyWriter0.setInternalSetting((Object) null, jsonInclude_Include0);
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = new PropertyName("virtual");
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      NameTransformer nameTransformer0 = NameTransformer.simpleTransformer("IWDW]", "{type: ");
      BeanPropertyWriter beanPropertyWriter2 = beanPropertyWriter1.rename(nameTransformer0);
      assertFalse(beanPropertyWriter2.willSuppressNulls());
      assertEquals("IWDW]virtual{type: ", beanPropertyWriter2.getName());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      NameTransformer nameTransformer1 = NameTransformer.chainedTransformer(nameTransformer0, nameTransformer0);
      BeanPropertyWriter beanPropertyWriter2 = beanPropertyWriter1.rename(nameTransformer1);
      assertSame(beanPropertyWriter2, beanPropertyWriter1);
      assertFalse(beanPropertyWriter2.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("A}|l{:RA}|l{:R", "A}|l{:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<String> jsonSerializer0 = (JsonSerializer<String>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "A}|l{:RA}|l{:R");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, asExternalTypeSerializer0, (JavaType) null, true, propertyName0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Can not override serializer
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertTrue(beanPropertyWriter0.hasSerializer());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      // Undeclared exception!
      try { 
        beanPropertyWriter0.assignNullSerializer((JsonSerializer<Object>) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Can not override null serializer
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      assertTrue(beanPropertyWriter0.hasNullSerializer());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      BeanPropertyWriter beanPropertyWriter1 = (BeanPropertyWriter)beanPropertyWriter0.readResolve();
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      beanPropertyWriter0.readResolve();
      assertTrue(beanPropertyWriter0.hasSerializer());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<Annotation> class0 = Annotation.class;
      beanPropertyWriter0.findAnnotation(class0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("A}|l{:RA}|l{:R", "A}|l{:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<String> jsonSerializer0 = (JsonSerializer<String>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "A}|l{:RA}|l{:R");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, asExternalTypeSerializer0, (JavaType) null, true, propertyName0);
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = beanPropertyWriter0.getContextAnnotation(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JsonFormat.Value jsonFormat_Value0 = beanPropertyWriter0.findFormatOverrides(annotationIntrospector0);
      assertNull(jsonFormat_Value0);
      
      JsonFormat.Value jsonFormat_Value1 = beanPropertyWriter0.findFormatOverrides(annotationIntrospector0);
      assertNull(jsonFormat_Value1);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JsonFormat.Value jsonFormat_Value0 = beanPropertyWriter0.findFormatOverrides((AnnotationIntrospector) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
      assertNull(jsonFormat_Value0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<Annotation> class0 = Annotation.class;
      beanPropertyWriter0.setInternalSetting((Object) null, class0);
      Object object0 = beanPropertyWriter0.getInternalSetting((Object) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getInternalSetting(beanPropertyWriter0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<Annotation> class0 = Annotation.class;
      beanPropertyWriter0.setInternalSetting(class0, class0);
      Object object0 = beanPropertyWriter0.setInternalSetting((Object) null, (Object) null);
      assertNull(object0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_EMPTY;
      beanPropertyWriter0.removeInternalSetting(jsonInclude_Include0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.setInternalSetting((Object) null, (Object) null);
      beanPropertyWriter0.removeInternalSetting((Object) null);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      beanPropertyWriter0.setInternalSetting((Object) null, dOMSerializer0);
      Object object0 = beanPropertyWriter0.removeInternalSetting("com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException");
      assertFalse(beanPropertyWriter0.willSuppressNulls());
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      boolean boolean0 = beanPropertyWriter0.hasSerializer();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.assignNullSerializer(defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      boolean boolean0 = beanPropertyWriter0.hasNullSerializer();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("A}|l{:RA}|l{:R", "A}|l{:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, false);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<String> jsonSerializer0 = (JsonSerializer<String>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "A}|l{:RA}|l{:R");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, asExternalTypeSerializer0, (JavaType) null, true, propertyName0);
      boolean boolean0 = beanPropertyWriter0.wouldConflictWithName(propertyName0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      PropertyName propertyName0 = new PropertyName(":~wkUfYi^YM:=F*Rfy");
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      PropertyName propertyName1 = PropertyName.USE_DEFAULT;
      boolean boolean0 = beanPropertyWriter1.wouldConflictWithName(propertyName1);
      assertFalse(boolean0);
      assertFalse(beanPropertyWriter1.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      beanPropertyWriter0.getRawSerializationType();
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      ObjectMapper objectMapper0 = new ObjectMapper();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, beanPropertyWriter0, false);
      Class<RoundingMode> class0 = RoundingMode.class;
      // Undeclared exception!
      try { 
        objectMapper0.convertValue((Object) iOContext0, class0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // Could not initialize class com.fasterxml.jackson.databind.JsonMappingException
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      ObjectMapper objectMapper0 = new ObjectMapper();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, beanPropertyWriter0, false);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, objectMapper0, (Writer) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      beanPropertyWriter0.serializeAsOmittedField(writerBasedJsonGenerator0, writerBasedJsonGenerator0, defaultSerializerProvider_Impl0);
      assertFalse(beanPropertyWriter0.willSuppressNulls());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      try { 
        beanPropertyWriter0.serializeAsElement((Object) null, (JsonGenerator) null, (SerializerProvider) null);
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
      try { 
        beanPropertyWriter0.serializeAsPlaceholder(beanPropertyWriter0, (JsonGenerator) null, (SerializerProvider) null);
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
      // Undeclared exception!
      try { 
        beanPropertyWriter0.serializeAsPlaceholder(defaultSerializerProvider_Impl0, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NoClassDefFoundError");
      
      } catch(NoClassDefFoundError e) {
         //
         // com.fasterxml.jackson.core.JsonProcessingException
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
      PropertyName propertyName0 = new PropertyName("~x|l^:RA}|l{R", "~x|l^:RA}|l{R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "~x|l^:RA}|l{R");
      PipedInputStream pipedInputStream0 = new PipedInputStream(65280);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(pipedInputStream0);
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, (JsonSerializer<?>) null, asExternalTypeSerializer0, (JavaType) null, true, pushbackInputStream0);
      JsonObjectFormatVisitor.Base jsonObjectFormatVisitor_Base0 = new JsonObjectFormatVisitor.Base();
      beanPropertyWriter0.depositSchemaProperty((JsonObjectFormatVisitor) jsonObjectFormatVisitor_Base0);
      assertFalse(beanPropertyWriter0.isVirtual());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("A}|l{:RA}|l{:R", "A}|l{:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "}7X");
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, mapType0, (JsonSerializer<?>) null, asExternalTypeSerializer0, mapType0, true, objectIdGenerators_IntSequenceGenerator0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = jsonNodeFactory0.objectNode();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("A}|l{:RA}|l{:R", "A}|l{:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, true);
      AnnotationMap annotationMap0 = new AnnotationMap();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "A}|l{:RA}|l{:R");
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      CoreXMLSerializers.XMLGregorianCalendarSerializer coreXMLSerializers_XMLGregorianCalendarSerializer0 = new CoreXMLSerializers.XMLGregorianCalendarSerializer();
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER, asExternalTypeSerializer0, (JavaType) null, true, coreXMLSerializers_XMLGregorianCalendarSerializer0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      beanPropertyWriter0.depositSchemaProperty(objectNode0, (SerializerProvider) defaultSerializerProvider_Impl0);
      assertFalse(objectNode0.isBigDecimal());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      beanPropertyWriter0._nonTrivialBaseType = (JavaType) simpleType0;
      PropertySerializerMap propertySerializerMap0 = PropertySerializerMap.emptyForRootValues();
      Class<Integer> class1 = Integer.TYPE;
      // Undeclared exception!
      try { 
        beanPropertyWriter0._findAndAddDynamic(propertySerializerMap0, class1, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter();
      try { 
        beanPropertyWriter0.get(beanPropertyWriter0);
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
      PropertyName propertyName0 = PropertyName.NO_NAME;
      BeanPropertyWriter beanPropertyWriter1 = new BeanPropertyWriter(beanPropertyWriter0, propertyName0);
      String string0 = beanPropertyWriter1.toString();
      assertFalse(beanPropertyWriter1.willSuppressNulls());
      assertEquals("property '' (virtual, no static serializer)", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Boolean boolean0 = Boolean.valueOf("Z}D3^:RA}|l{:R");
      PropertyName propertyName0 = new PropertyName("Z}D3^:RA}|l{:R", "Z}D3^:RA}|l{:R");
      POJOPropertyBuilder pOJOPropertyBuilder0 = new POJOPropertyBuilder(propertyName0, (AnnotationIntrospector) null, (boolean) boolean0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      JsonSerializer<String> jsonSerializer0 = (JsonSerializer<String>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, (AnnotatedMember) null, propertyMetadata0);
      AsExternalTypeSerializer asExternalTypeSerializer0 = new AsExternalTypeSerializer(classNameIdResolver0, beanProperty_Std0, "Z}D3^:RA}|l{:R");
      BeanPropertyWriter beanPropertyWriter0 = new BeanPropertyWriter(pOJOPropertyBuilder0, (AnnotatedMember) null, annotationMap0, (JavaType) null, jsonSerializer0, asExternalTypeSerializer0, (JavaType) null, (boolean) boolean0, propertyName0);
      String string0 = beanPropertyWriter0.toString();
      assertEquals("property 'Z}D3^:RA}|l{:R' (virtual, static serializer of type com.fasterxml.jackson.databind.JsonSerializer$MockitoMock$334517246)", string0);
  }
}
