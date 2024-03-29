/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:07:12 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.ObjectIdResolver;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.annotation.JsonAppend;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.impl.AttributePropertyWriter;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
import java.lang.reflect.Type;
import java.time.chrono.HijrahEra;
import java.util.ArrayList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JacksonAnnotationIntrospector_ESTest extends JacksonAnnotationIntrospector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructNoTypeResolverBuilder();
      assertFalse(stdTypeResolverBuilder0.isTypeIdVisible());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructStdTypeResolverBuilder();
      assertFalse(stdTypeResolverBuilder0.isTypeIdVisible());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<AttributePropertyWriter> class0 = AttributePropertyWriter.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Object object0 = jacksonAnnotationIntrospector0.findFilterId(annotatedClass0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.findFilterId((Annotated) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.AnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Boolean boolean0 = jacksonAnnotationIntrospector0.isTypeId(annotatedConstructor0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Version version0 = jacksonAnnotationIntrospector0.version();
      assertEquals(4, version0.getPatchLevel());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      DeserializationFeature deserializationFeature0 = DeserializationFeature.FAIL_ON_NUMBERS_FOR_ENUMS;
      String string0 = jacksonAnnotationIntrospector0.findEnumValue(deserializationFeature0);
      assertEquals("FAIL_ON_NUMBERS_FOR_ENUMS", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      PropertyName propertyName0 = jacksonAnnotationIntrospector0.findRootName(annotatedClass0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      String[] stringArray0 = jacksonAnnotationIntrospector0.findPropertiesToIgnore((Annotated) annotatedConstructor0);
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<ObjectIdResolver> class0 = ObjectIdResolver.class;
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructType((Type) class0, (Class<?>) class0);
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.findPropertyContentTypeResolver(deserializationConfig0, (AnnotatedMember) null, javaType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must call method with a container type (got [simple type, class com.fasterxml.jackson.annotation.ObjectIdResolver])
         //
         verifyException("com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      List<NamedType> list0 = jacksonAnnotationIntrospector0.findSubtypes(annotatedConstructor0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<NamedType> class0 = NamedType.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      String string0 = jacksonAnnotationIntrospector0.findTypeName(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      ObjectIdInfo objectIdInfo0 = jacksonAnnotationIntrospector0.findObjectReferenceInfo(annotatedConstructor0, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findKeySerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findContentSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findNullSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_NULL;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusion(annotatedConstructor0, jsonInclude_Include0);
      assertEquals(JsonInclude.Include.NON_NULL, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.USE_DEFAULTS;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusionForContent(annotatedConstructor0, jsonInclude_Include0);
      assertEquals(JsonInclude.Include.USE_DEFAULTS, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Value jsonInclude_Value0 = jacksonAnnotationIntrospector0.findPropertyInclusion(annotatedConstructor0);
      assertEquals(JsonInclude.Include.USE_DEFAULTS, jsonInclude_Value0.getValueInclusion());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationType(annotatedConstructor0);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationKeyType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationContentType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      jacksonAnnotationIntrospector0.findSerializationTyping(annotatedConstructor0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findSerializationConverter(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findSerializationContentConverter(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      ArrayList<BeanPropertyWriter> arrayList0 = new ArrayList<BeanPropertyWriter>();
      Class<NamedType> class0 = NamedType.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, deserializationConfig0);
      jacksonAnnotationIntrospector0.findAndAddVirtualProperties(deserializationConfig0, annotatedClass0, arrayList0);
      assertEquals(0, arrayList0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonAppend_Prop0).name();
      doReturn((String) null).when(jsonAppend_Prop0).namespace();
      doReturn(false).when(jsonAppend_Prop0).required();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, deserializationConfig0);
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, deserializationConfig0, annotatedClass0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, stdSubtypeResolver0, simpleMixInResolver0, rootNameLookup0);
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn("").when(jsonAppend_Prop0).name();
      doReturn("").when(jsonAppend_Prop0).namespace();
      doReturn(true).when(jsonAppend_Prop0).required();
      doReturn((Class) null).when(jsonAppend_Prop0).type();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, deserializationConfig0);
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, deserializationConfig0, annotatedClass0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.cfg.MapperConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findKeyDeserializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<HijrahEra> class0 = HijrahEra.class;
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, (Class<?>) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Object> class0 = Object.class;
      Class<BeanPropertyWriter> class1 = BeanPropertyWriter.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertEquals("class java.lang.Object", class2.toString());
      assertNotNull(class2);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("kC", "kC");
      assertTrue(propertyName0.hasNamespace());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("Attempted to unwrap single value array for single 'long' value but there was more than a single value in the array", (String) null);
      assertFalse(propertyName0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector", "");
      assertFalse(propertyName0.hasNamespace());
      assertFalse(propertyName0.isEmpty());
  }
}
