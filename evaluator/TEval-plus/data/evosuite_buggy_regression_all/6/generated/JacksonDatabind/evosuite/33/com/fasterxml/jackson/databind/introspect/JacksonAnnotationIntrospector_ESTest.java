/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:23:30 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.annotation.JsonAppend;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.BasicClassIntrospector;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.type.SimpleType;
import java.util.LinkedList;
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<Object> class0 = Object.class;
      NamedType namedType0 = new NamedType(class0, "");
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(namedType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.findFilterId((AnnotatedClass) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.AnnotationIntrospector", e);
      }
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
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.isTypeId((AnnotatedMember) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.AnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(basicBeanDescription0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Version version0 = jacksonAnnotationIntrospector0.version();
      assertEquals(2, version0.getMajorVersion());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      SerializationFeature serializationFeature0 = SerializationFeature.WRITE_ENUMS_USING_INDEX;
      String string0 = jacksonAnnotationIntrospector0.findEnumValue(serializationFeature0);
      assertEquals("WRITE_ENUMS_USING_INDEX", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      PropertyName propertyName0 = jacksonAnnotationIntrospector0.findRootName(annotatedClass0);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      String[] stringArray0 = jacksonAnnotationIntrospector0.findPropertiesToIgnore((Annotated) annotatedConstructor0);
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Object object0 = jacksonAnnotationIntrospector0._findFilterId(annotatedConstructor0);
      assertNull(object0);
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
      Class<PropertyName> class0 = PropertyName.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
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
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
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
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_DEFAULT;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusion(annotatedConstructor0, jsonInclude_Include0);
      assertEquals(JsonInclude.Include.NON_DEFAULT, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_ABSENT;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusionForContent(annotatedConstructor0, jsonInclude_Include0);
      assertSame(jsonInclude_Include1, jsonInclude_Include0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonInclude.Value jsonInclude_Value0 = jacksonAnnotationIntrospector0.findPropertyInclusion(annotatedConstructor0);
      assertEquals(JsonInclude.Include.USE_DEFAULTS, jsonInclude_Value0.getContentInclusion());
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
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
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
      LinkedList<BeanPropertyWriter> linkedList0 = new LinkedList<BeanPropertyWriter>();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      jacksonAnnotationIntrospector0.findAndAddVirtualProperties((MapperConfig<?>) null, annotatedClass0, linkedList0);
      assertEquals(0, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.ALWAYS;
      JsonAppend.Attr jsonAppend_Attr0 = mock(JsonAppend.Attr.class, new ViolatedAssumptionAnswer());
      doReturn(jsonInclude_Include0).when(jsonAppend_Attr0).include();
      doReturn("p[+").when(jsonAppend_Attr0).propName();
      doReturn("p[+").when(jsonAppend_Attr0).propNamespace();
      doReturn(false).when(jsonAppend_Attr0).required();
      doReturn("p[+").when(jsonAppend_Attr0).value();
      Class<AnnotatedClass> class1 = AnnotatedClass.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      BeanPropertyWriter beanPropertyWriter0 = jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Attr0, (MapperConfig<?>) null, annotatedClass0, simpleType0);
      assertFalse(beanPropertyWriter0.isRequired());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<JsonSerializer> class1 = JsonSerializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      JsonAppend.Attr jsonAppend_Attr0 = mock(JsonAppend.Attr.class, new ViolatedAssumptionAnswer());
      doReturn((JsonInclude.Include) null).when(jsonAppend_Attr0).include();
      doReturn("").when(jsonAppend_Attr0).propName();
      doReturn("").when(jsonAppend_Attr0).propNamespace();
      doReturn(true).when(jsonAppend_Attr0).required();
      doReturn("}Ud^-%OY$0Q]").when(jsonAppend_Attr0).value();
      BeanPropertyWriter beanPropertyWriter0 = jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Attr0, (MapperConfig<?>) null, annotatedClass0, simpleType0);
      assertEquals("}Ud^-%OY$0Q]", beanPropertyWriter0.getName());
      assertTrue(beanPropertyWriter0.isRequired());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonAppend_Prop0).name();
      doReturn((String) null).when(jsonAppend_Prop0).namespace();
      doReturn(false).when(jsonAppend_Prop0).required();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, (MapperConfig<?>) null, annotatedClass0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.LONG_DESC;
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      AnnotatedClass annotatedClass0 = beanSerializerBuilder0.getClassInfo();
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn("O]_`Bqbn_k<Z2d").when(jsonAppend_Prop0).name();
      doReturn("zA<hLWe+7SF~/ctbQ6").when(jsonAppend_Prop0).namespace();
      doReturn(true).when(jsonAppend_Prop0).required();
      doReturn((Class) null).when(jsonAppend_Prop0).type();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, (MapperConfig<?>) null, annotatedClass0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findKeyDeserializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Object> class0 = Object.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<Object> class0 = Object.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      Class<NamedType> class1 = NamedType.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertEquals("class com.fasterxml.jackson.databind.introspect.AnnotatedClass", class2.toString());
      assertNotNull(class2);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName(".;k$5>S~J}wBET=?", (String) null);
      assertEquals(".;k$5>S~J}wBET=?", propertyName0.getSimpleName());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("71LtWEG'\u0000v5),)4bq", "");
      assertFalse(propertyName0.hasNamespace());
      assertFalse(propertyName0.isEmpty());
  }
}