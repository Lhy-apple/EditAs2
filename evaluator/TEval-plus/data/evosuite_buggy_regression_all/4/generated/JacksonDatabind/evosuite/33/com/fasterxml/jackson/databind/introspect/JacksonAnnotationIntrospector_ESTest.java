/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:41:22 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.annotation.JsonAppend;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.cfg.BaseSettings;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.SimpleMixInResolver;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.SubtypeResolver;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.impl.AttributePropertyWriter;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.RootNameLookup;
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
      assertNull(stdTypeResolverBuilder0.getTypeProperty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructStdTypeResolverBuilder();
      assertNull(stdTypeResolverBuilder0.getTypeProperty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      SimpleType simpleType0 = SimpleType.construct(class0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedMethod> class0 = AnnotatedMethod.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Object object0 = jacksonAnnotationIntrospector0.findFilterId(annotatedClass0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findFilterId((Annotated) annotatedConstructor0);
      assertNull(object0);
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
      assertEquals(2, version0.getMajorVersion());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.USE_DEFAULTS;
      String string0 = jacksonAnnotationIntrospector0.findEnumValue(jsonInclude_Include0);
      assertEquals("USE_DEFAULTS", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, (SubtypeResolver) null, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, deserializationConfig0);
      PropertyName propertyName0 = jacksonAnnotationIntrospector0.findRootName(annotatedClass0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      String[] stringArray0 = jacksonAnnotationIntrospector0.findPropertiesToIgnore((Annotated) annotatedConstructor0);
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      List<NamedType> list0 = jacksonAnnotationIntrospector0.findSubtypes(annotatedConstructor0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      String string0 = jacksonAnnotationIntrospector0.findTypeName(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      ObjectIdInfo objectIdInfo0 = jacksonAnnotationIntrospector0.findObjectReferenceInfo(annotatedConstructor0, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findKeySerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findContentSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findNullSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_DEFAULT;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusion(annotatedConstructor0, jsonInclude_Include0);
      assertSame(jsonInclude_Include1, jsonInclude_Include0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_ABSENT;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusionForContent(annotatedConstructor0, jsonInclude_Include0);
      assertSame(jsonInclude_Include0, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonInclude.Value jsonInclude_Value0 = jacksonAnnotationIntrospector0.findPropertyInclusion(annotatedConstructor0);
      assertEquals(JsonInclude.Include.USE_DEFAULTS, jsonInclude_Value0.getValueInclusion());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationType(annotatedConstructor0);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationKeyType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationContentType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      jacksonAnnotationIntrospector0.findSerializationTyping(annotatedConstructor0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findSerializationConverter(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findSerializationContentConverter(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Integer> class0 = Integer.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, (SubtypeResolver) null, simpleMixInResolver0, rootNameLookup0);
      jacksonAnnotationIntrospector0.findAndAddVirtualProperties(deserializationConfig0, annotatedClass0, (List<BeanPropertyWriter>) null);
      assertEquals(15214880, deserializationConfig0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedMethod> class0 = AnnotatedMethod.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonAppend.Attr jsonAppend_Attr0 = mock(JsonAppend.Attr.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonAppend_Attr0).propName();
      doReturn((String) null).when(jsonAppend_Attr0).propNamespace();
      doReturn(false).when(jsonAppend_Attr0).required();
      doReturn((String) null).when(jsonAppend_Attr0).value();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Attr0, (MapperConfig<?>) null, annotatedClass0, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Integer> class0 = Integer.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, (SubtypeResolver) null, simpleMixInResolver0, rootNameLookup0);
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_DEFAULT;
      JsonAppend.Attr jsonAppend_Attr0 = mock(JsonAppend.Attr.class, new ViolatedAssumptionAnswer());
      doReturn(jsonInclude_Include0).when(jsonAppend_Attr0).include();
      doReturn("DRHe@I").when(jsonAppend_Attr0).propName();
      doReturn("DRHe@I").when(jsonAppend_Attr0).propNamespace();
      doReturn(true).when(jsonAppend_Attr0).required();
      doReturn("DRHe@I").when(jsonAppend_Attr0).value();
      JavaType javaType0 = TypeFactory.unknownType();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Attr0, deserializationConfig0, annotatedClass0, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.cfg.MapperConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Integer> class0 = Integer.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, (SubtypeResolver) null, simpleMixInResolver0, rootNameLookup0);
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonAppend_Prop0).name();
      doReturn((String) null).when(jsonAppend_Prop0).namespace();
      doReturn(false).when(jsonAppend_Prop0).required();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, deserializationConfig0, annotatedClass0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      SimpleMixInResolver simpleMixInResolver0 = new SimpleMixInResolver((ClassIntrospector.MixInResolver) null);
      RootNameLookup rootNameLookup0 = new RootNameLookup();
      DeserializationConfig deserializationConfig0 = new DeserializationConfig((BaseSettings) null, (SubtypeResolver) null, simpleMixInResolver0, rootNameLookup0);
      Class<String> class0 = String.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, deserializationConfig0);
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn("R!T-V%9g<*;").when(jsonAppend_Prop0).name();
      doReturn("R!T-V%9g<*;").when(jsonAppend_Prop0).namespace();
      doReturn(true).when(jsonAppend_Prop0).required();
      doReturn(class0).when(jsonAppend_Prop0).type();
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
  public void test32()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findKeyDeserializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, (Class<?>) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AttributePropertyWriter> class0 = AttributePropertyWriter.class;
      Class<String> class1 = String.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertNotNull(class2);
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<String> class0 = String.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("", "");
      assertEquals("", propertyName0.getSimpleName());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName(",}", (String) null);
      assertEquals(",}", propertyName0.getSimpleName());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("}1t&J;nB9", "");
      assertFalse(propertyName0.hasNamespace());
      assertEquals("}1t&J;nB9", propertyName0.getSimpleName());
  }
}
