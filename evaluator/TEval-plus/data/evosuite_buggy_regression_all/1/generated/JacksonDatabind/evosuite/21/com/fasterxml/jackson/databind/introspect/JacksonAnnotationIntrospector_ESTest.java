/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:34:48 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.annotation.JsonAppend;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
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
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdTypeResolverBuilder;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import java.io.BufferedInputStream;
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
      assertFalse(stdTypeResolverBuilder0.isTypeIdVisible());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertNotSame(objectReader0, objectReader1);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<BufferedInputStream> class0 = BufferedInputStream.class;
      boolean boolean0 = objectMapper0.canSerialize(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Boolean boolean0 = jacksonAnnotationIntrospector0.isTypeId(annotatedConstructor0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Version version0 = jacksonAnnotationIntrospector0.version();
      assertFalse(version0.isUknownVersion());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0.findSerializationSortAlphabetically((AnnotatedClass) null);
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
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<BeanPropertyWriter> class0 = BeanPropertyWriter.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      String string0 = jacksonAnnotationIntrospector0.findTypeName(annotatedClass0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedClass> class0 = AnnotatedClass.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
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
      Class<NamedType> class0 = NamedType.class;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.reader();
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertNotSame(objectReader1, objectReader0);
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
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      ObjectIdInfo objectIdInfo0 = jacksonAnnotationIntrospector0.findObjectReferenceInfo(annotatedConstructor0, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findKeySerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findContentSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findNullSerializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.ALWAYS;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusion(annotatedConstructor0, jsonInclude_Include0);
      assertEquals(JsonInclude.Include.ALWAYS, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_DEFAULT;
      JsonInclude.Include jsonInclude_Include1 = jacksonAnnotationIntrospector0.findSerializationInclusionForContent(annotatedConstructor0, jsonInclude_Include0);
      assertSame(jsonInclude_Include0, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationKeyType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Class<?> class0 = jacksonAnnotationIntrospector0.findSerializationContentType(annotatedConstructor0, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      jacksonAnnotationIntrospector0.findSerializationTyping(annotatedConstructor0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findSerializationContentConverter(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(jsonAppend_Prop0).name();
      doReturn((String) null).when(jsonAppend_Prop0).namespace();
      doReturn(false).when(jsonAppend_Prop0).required();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, (MapperConfig<?>) null, (AnnotatedClass) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      JsonAppend.Prop jsonAppend_Prop0 = mock(JsonAppend.Prop.class, new ViolatedAssumptionAnswer());
      doReturn("rx&m:i").when(jsonAppend_Prop0).name();
      doReturn("rx&m:i").when(jsonAppend_Prop0).namespace();
      doReturn(true).when(jsonAppend_Prop0).required();
      doReturn((Class) null).when(jsonAppend_Prop0).type();
      // Undeclared exception!
      try { 
        jacksonAnnotationIntrospector0._constructVirtualProperty(jsonAppend_Prop0, (MapperConfig<?>) null, (AnnotatedClass) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      BasicBeanDescription basicBeanDescription0 = BasicClassIntrospector.STRING_DESC;
      AnnotatedConstructor annotatedConstructor0 = basicBeanDescription0.findDefaultConstructor();
      Object object0 = jacksonAnnotationIntrospector0.findKeyDeserializer(annotatedConstructor0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<NamedType> class0 = NamedType.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, jacksonAnnotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = jacksonAnnotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit((Class<?>) null, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<ObjectIdGenerators.UUIDGenerator> class0 = ObjectIdGenerators.UUIDGenerator.class;
      Class<?> class1 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      Class<AnnotatedMethod> class0 = AnnotatedMethod.class;
      Class<Object> class1 = Object.class;
      Class<?> class2 = jacksonAnnotationIntrospector0._classIfExplicit(class0, class1);
      assertNotNull(class2);
      assertEquals("class com.fasterxml.jackson.databind.introspect.AnnotatedMethod", class2.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("", "");
      assertTrue(propertyName0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("OtpKghf]fLo:t", (String) null);
      assertTrue(propertyName0.hasSimpleName());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      JacksonAnnotationIntrospector jacksonAnnotationIntrospector0 = new JacksonAnnotationIntrospector();
      PropertyName propertyName0 = jacksonAnnotationIntrospector0._propertyName("*2|W\u0007S-;F\"46", "");
      assertFalse(propertyName0.hasNamespace());
      assertTrue(propertyName0.hasSimpleName());
  }
}