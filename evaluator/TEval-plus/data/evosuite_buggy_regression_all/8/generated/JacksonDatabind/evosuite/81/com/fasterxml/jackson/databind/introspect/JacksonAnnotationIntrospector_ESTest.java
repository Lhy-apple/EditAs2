/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:17:32 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.TypeResolutionContext;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
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
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationType((Annotated) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(basicBeanDescription0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructNoTypeResolverBuilder();
      assertNull(stdTypeResolverBuilder0.getTypeProperty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      StdTypeResolverBuilder stdTypeResolverBuilder0 = jacksonAnnotationIntrospector0._constructStdTypeResolverBuilder();
      assertFalse(stdTypeResolverBuilder0.isTypeIdVisible());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedConstructor> class0 = AnnotatedConstructor.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, "@lNY`@EYrN 1!Ocq", (JavaType) null);
      Class<?> class1 = jacksonAnnotationIntrospector0.findDeserializationContentType(virtualAnnotatedMember0, (JavaType) null);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature mapperFeature0 = MapperFeature.USE_WRAPPER_NAME_AS_PROPERTY_NAME;
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(mapperFeature0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BasicBeanDescription> class0 = BasicBeanDescription.class;
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      objectMapper0.acceptJsonFormatVisitor((Class<?>) class0, (JsonFormatVisitorWrapper) jsonFormatVisitorWrapper_Base0);
      assertEquals(0, objectMapper0.mixInCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.INT_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<?> class0 = jacksonAnnotationIntrospector0.findDeserializationType(annotatedClass0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedConstructor> class0 = AnnotatedConstructor.class;
      JavaType javaType0 = TypeFactory.unknownType();
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[2] = javaType0;
      javaTypeArray0[4] = javaType0;
      MapType mapType0 = MapType.construct(class0, (TypeBindings) null, javaType0, javaTypeArray0, javaTypeArray0[2], javaTypeArray0[4]);
      Class<?> class1 = jacksonAnnotationIntrospector0.findSerializationKeyType((Annotated) null, mapType0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
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
  public void test10()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<SimpleObjectIdResolver> class0 = SimpleObjectIdResolver.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, ".", (JavaType) null);
      Class<?> class1 = jacksonAnnotationIntrospector0.findDeserializationKeyType(virtualAnnotatedMember0, (JavaType) null);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Version version0 = jacksonAnnotationIntrospector0.version();
      assertTrue(version0.isSnapshot());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
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
  public void test13()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationContentType(annotatedClass0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0.readResolve();
      assertSame(object0, jacksonAnnotationIntrospector0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.setAnnotationIntrospector(jacksonAnnotationIntrospector0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jacksonAnnotationIntrospector0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      SerializationFeature serializationFeature0 = SerializationFeature.FAIL_ON_UNWRAPPED_TYPE_IDENTIFIERS;
      String string0 = jacksonAnnotationIntrospector0.findEnumValue(serializationFeature0);
      assertEquals("FAIL_ON_UNWRAPPED_TYPE_IDENTIFIERS", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0.findRootName(annotatedClass0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      String string0 = jacksonAnnotationIntrospector0.findClassDescription(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, "MIMbE", (JavaType) null);
      Object object0 = jacksonAnnotationIntrospector0.findInjectableValueId(virtualAnnotatedMember0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(objectMapper0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedConstructor> class0 = AnnotatedConstructor.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember((TypeResolutionContext) null, class0, "@lNY`@EYrN 1!Ocq", (JavaType) null);
      List<NamedType> list0 = jacksonAnnotationIntrospector0.findSubtypes(virtualAnnotatedMember0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      String string0 = jacksonAnnotationIntrospector0.findTypeName(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      ObjectIdInfo objectIdInfo0 = jacksonAnnotationIntrospector0.findObjectReferenceInfo(annotatedClass0, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      Object object0 = jacksonAnnotationIntrospector0.findKeySerializer(annotatedClass0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Object object0 = new Object();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(object0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.BOOLEAN_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      boolean boolean0 = jacksonAnnotationIntrospector0.hasCreatorAnnotation(annotatedClass0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector1 = jacksonAnnotationIntrospector0.setConstructorPropertiesImpliesCreator(false);
      boolean boolean0 = jacksonAnnotationIntrospector1.hasCreatorAnnotation(annotatedClass0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      jacksonAnnotationIntrospector0.findCreatorBinding(annotatedClass0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonFactory jsonFactory0 = new JsonFactory((ObjectCodec) null);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector1 = jacksonAnnotationIntrospector0.setConstructorPropertiesImpliesCreator(false);
      objectMapper0.setAnnotationIntrospectors(jacksonAnnotationIntrospector0, jacksonAnnotationIntrospector1);
      JsonInclude jsonInclude0 = mock(JsonInclude.class, new ViolatedAssumptionAnswer());
      doReturn((JsonInclude.Include) null).when(jsonInclude0).content();
      doReturn((Class) null).when(jsonInclude0).contentFilter();
      doReturn((JsonInclude.Include) null).when(jsonInclude0).value();
      doReturn((Class) null).when(jsonInclude0).valueFilter();
      JsonInclude.Value jsonInclude_Value0 = JsonInclude.Value.from(jsonInclude0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(jsonInclude_Value0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      MapperFeature[] mapperFeatureArray0 = new MapperFeature[9];
      MapperFeature mapperFeature0 = MapperFeature.AUTO_DETECT_IS_GETTERS;
      mapperFeatureArray0[0] = mapperFeature0;
      MapperFeature mapperFeature1 = MapperFeature.INFER_CREATOR_FROM_CONSTRUCTOR_PROPERTIES;
      mapperFeatureArray0[1] = mapperFeature1;
      mapperFeatureArray0[2] = mapperFeatureArray0[0];
      mapperFeatureArray0[3] = mapperFeatureArray0[0];
      mapperFeatureArray0[4] = mapperFeature0;
      mapperFeatureArray0[5] = mapperFeatureArray0[2];
      mapperFeatureArray0[6] = mapperFeatureArray0[1];
      mapperFeatureArray0[7] = mapperFeatureArray0[5];
      mapperFeatureArray0[8] = mapperFeatureArray0[6];
      objectMapper0.disable(mapperFeatureArray0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(mapperFeatureArray0[3]);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedParameter> class0 = AnnotatedParameter.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<MapperFeature> class0 = MapperFeature.class;
      Class<AnnotatedClass> class1 = AnnotatedClass.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertNotNull(class2);
      assertEquals("class com.fasterxml.jackson.databind.MapperFeature", class2.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<MapperFeature> class0 = MapperFeature.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("f", "f");
      assertTrue(propertyName0.hasNamespace());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("", "");
      assertFalse(propertyName0.hasSimpleName());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName(";0w6[`DWGOze", (String) null);
      assertFalse(propertyName0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", "");
      assertFalse(propertyName0.hasNamespace());
      assertEquals("com.fasterxml.jackson.databind.ser.BeanPropertyWriter", propertyName0.getSimpleName());
  }
}
