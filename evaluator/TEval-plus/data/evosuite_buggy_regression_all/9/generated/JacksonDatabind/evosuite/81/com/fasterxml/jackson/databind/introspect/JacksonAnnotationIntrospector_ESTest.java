/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:55:18 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.lang.annotation.Annotation;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JacksonAnnotationIntrospector_ESTest extends JacksonAnnotationIntrospector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationType((Annotated) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructNoTypeResolverBuilder();
      assertNull(stdTypeResolverBuilder0.getTypeProperty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructStdTypeResolverBuilder();
      assertFalse(stdTypeResolverBuilder0.isTypeIdVisible());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findDeserializationContentType(annotatedClass0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      ObjectWriter objectWriter0 = objectMapper0.writerFor(class0);
      assertTrue(objectWriter0.hasPrefetchedSerializer());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Class<?> class1 = jacksonAnnotationIntrospector0.findDeserializationType(annotatedClass0, simpleType0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationKeyType((Annotated) null, javaType0);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.hasAnySetterAnnotation((AnnotatedMethod) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.AnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<?> class0 = jacksonAnnotationIntrospector0.findDeserializationKeyType(annotatedClass0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Version version0 = jacksonAnnotationIntrospector0.version();
      assertFalse(version0.isUknownVersion());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.hasAnyGetterAnnotation((AnnotatedMethod) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.AnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationContentType((Annotated) null, collectionLikeType0);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector1 = jacksonAnnotationIntrospector0.setConstructorPropertiesImpliesCreator(false);
      jacksonAnnotationIntrospector1._annotationsInside = null;
      Object object0 = jacksonAnnotationIntrospector1.readResolve();
      assertSame(object0, jacksonAnnotationIntrospector0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      MapperFeature mapperFeature0 = MapperFeature.AUTO_DETECT_IS_GETTERS;
      String string0 = jacksonAnnotationIntrospector0.findEnumValue(mapperFeature0);
      assertEquals("AUTO_DETECT_IS_GETTERS", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0.findRootName(annotatedClass0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      String string0 = jacksonAnnotationIntrospector0.findClassDescription(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<AnnotatedParameter> class0 = AnnotatedParameter.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, "", (JavaType) null);
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findInjectableValueId(virtualAnnotatedMember0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(objectMapper0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      List<NamedType> list0 = jacksonAnnotationIntrospector0.findSubtypes(annotatedClass0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = jacksonAnnotationIntrospector0.findTypeName(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      ObjectIdInfo objectIdInfo0 = jacksonAnnotationIntrospector0.findObjectReferenceInfo(annotatedClass0, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.findKeySerializer(annotatedClass0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(beanSerializerBuilder0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      boolean boolean0 = jacksonAnnotationIntrospector0.hasCreatorAnnotation(annotatedClass0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector1 = jacksonAnnotationIntrospector0.setConstructorPropertiesImpliesCreator(false);
      boolean boolean0 = jacksonAnnotationIntrospector1.hasCreatorAnnotation(annotatedClass0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      jacksonAnnotationIntrospector0.findCreatorBinding(annotatedClass0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector1 = jacksonAnnotationIntrospector0.setConstructorPropertiesImpliesCreator(false);
      objectMapper0.setAnnotationIntrospectors(jacksonAnnotationIntrospector1, jacksonAnnotationIntrospector0);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(pipedInputStream0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonGenerator.Feature jsonGenerator_Feature0 = JsonGenerator.Feature.QUOTE_NON_NUMERIC_NUMBERS;
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[8];
      MapperFeature mapperFeature0 = MapperFeature.AUTO_DETECT_GETTERS;
      mapperFeatureArray0[0] = mapperFeature0;
      MapperFeature mapperFeature1 = MapperFeature.INFER_CREATOR_FROM_CONSTRUCTOR_PROPERTIES;
      mapperFeatureArray0[1] = mapperFeature1;
      mapperFeatureArray0[2] = mapperFeatureArray0[1];
      mapperFeatureArray0[3] = mapperFeatureArray0[1];
      mapperFeatureArray0[4] = mapperFeature0;
      mapperFeatureArray0[5] = mapperFeatureArray0[2];
      mapperFeatureArray0[6] = mapperFeatureArray0[2];
      mapperFeatureArray0[7] = mapperFeature0;
      ObjectMapper objectMapper1 = objectMapper0.disable(mapperFeatureArray0);
      ObjectReader objectReader0 = objectMapper1.readerForUpdating(jsonGenerator_Feature0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Annotation> class0 = Annotation.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Object> class0 = Object.class;
      Class<DeserializationFeature> class1 = DeserializationFeature.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertNotNull(class2);
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("Can not add mapping from class ", "Can not add mapping from class ");
      assertTrue(propertyName0.hasNamespace());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("", "");
      assertEquals("", propertyName0.getSimpleName());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("'", (String) null);
      assertFalse(propertyName0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("Failed to narrow type %s with annotation (value %s), from '%s': %s", "");
      assertFalse(propertyName0.hasNamespace());
      assertEquals("Failed to narrow type %s with annotation (value %s), from '%s': %s", propertyName0.getSimpleName());
  }
}