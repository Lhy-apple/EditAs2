/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:42:48 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.annotation.JsonPOJOBuilder;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.introspect.Annotated;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnnotationIntrospector_ESTest extends AnnotationIntrospector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty.Type annotationIntrospector_ReferenceProperty_Type0 = AnnotationIntrospector.ReferenceProperty.Type.BACK_REFERENCE;
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = new AnnotationIntrospector.ReferenceProperty(annotationIntrospector_ReferenceProperty_Type0, "r\"F");
      AnnotationIntrospector.ReferenceProperty.Type annotationIntrospector_ReferenceProperty_Type1 = annotationIntrospector_ReferenceProperty0.getType();
      assertEquals(AnnotationIntrospector.ReferenceProperty.Type.BACK_REFERENCE, annotationIntrospector_ReferenceProperty_Type1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = AnnotationIntrospector.ReferenceProperty.back("Can not refine serialization key type %s into %s; types not related");
      String string0 = annotationIntrospector_ReferenceProperty0.getName();
      assertEquals("Can not refine serialization key type %s into %s; types not related", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = AnnotationIntrospector.ReferenceProperty.managed((String) null);
      boolean boolean0 = annotationIntrospector_ReferenceProperty0.isBackReference();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Object object0 = annotationIntrospector0.findContentSerializer((Annotated) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      objectMapper0.setAnnotationIntrospectors(annotationIntrospector0, annotationIntrospector0);
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      ObjectWriter objectWriter0 = objectMapper0.writerFor(class0);
      assertTrue(objectWriter0.hasPrefetchedSerializer());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<?> class0 = annotationIntrospector0.findSerializationKeyType((Annotated) null, (JavaType) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      String string0 = annotationIntrospector0.findClassDescription((AnnotatedClass) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.setAnnotationIntrospector(annotationIntrospector0);
      Object object0 = new Object();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(object0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      PropertyName propertyName0 = annotationIntrospector0.findRootName((AnnotatedClass) null);
      assertNull(propertyName0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<Object> class0 = Object.class;
      AnnotatedClass annotatedClass0 = AnnotatedClass.constructWithoutSuperTypes((Class<?>) class0, (MapperConfig<?>) null);
      JsonPOJOBuilder.Value jsonPOJOBuilder_Value0 = annotationIntrospector0.findPOJOBuilderConfig(annotatedClass0);
      assertNull(jsonPOJOBuilder_Value0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Object object0 = annotationIntrospector0.findKeySerializer((Annotated) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectMapper objectMapper1 = objectMapper0.setAnnotationIntrospector(annotationIntrospector0);
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(objectMapper1);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector annotationIntrospector1 = AnnotationIntrospector.pair(annotationIntrospector0, annotationIntrospector0);
      assertNotNull(annotationIntrospector1);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      String string0 = annotationIntrospector0.findTypeName((AnnotatedClass) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Object object0 = annotationIntrospector0.findSerializationContentConverter((AnnotatedMember) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      ObjectIdInfo objectIdInfo0 = annotationIntrospector0.findObjectReferenceInfo((Annotated) null, (ObjectIdInfo) null);
      assertNull(objectIdInfo0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      List<NamedType> list0 = annotationIntrospector0.findSubtypes((Annotated) null);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.ALWAYS;
      JsonInclude.Include jsonInclude_Include1 = annotationIntrospector0.findSerializationInclusion((Annotated) null, jsonInclude_Include0);
      assertSame(jsonInclude_Include0, jsonInclude_Include1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      ArrayList<AnnotationIntrospector> arrayList0 = new ArrayList<AnnotationIntrospector>();
      Collection<AnnotationIntrospector> collection0 = annotationIntrospector0.allIntrospectors((Collection<AnnotationIntrospector>) arrayList0);
      assertTrue(collection0.contains(annotationIntrospector0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Object object0 = annotationIntrospector0.findDeserializationContentConverter((AnnotatedMember) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Collection<AnnotationIntrospector> collection0 = annotationIntrospector0.allIntrospectors();
      assertTrue(collection0.contains(annotationIntrospector0));
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      JsonInclude.Include jsonInclude_Include0 = JsonInclude.Include.NON_ABSENT;
      JsonInclude.Include jsonInclude_Include1 = annotationIntrospector0.findSerializationInclusionForContent((Annotated) null, jsonInclude_Include0);
      assertSame(jsonInclude_Include1, jsonInclude_Include0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      String[] stringArray0 = annotationIntrospector0.findPropertiesToIgnore((Annotated) null);
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = AnnotationIntrospector.ReferenceProperty.back("W$g%0lnXIzj6clc");
      boolean boolean0 = annotationIntrospector_ReferenceProperty0.isManagedReference();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty.Type annotationIntrospector_ReferenceProperty_Type0 = AnnotationIntrospector.ReferenceProperty.Type.MANAGED_REFERENCE;
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = new AnnotationIntrospector.ReferenceProperty(annotationIntrospector_ReferenceProperty_Type0, ">c-S");
      boolean boolean0 = annotationIntrospector_ReferenceProperty0.isManagedReference();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      AnnotationIntrospector.ReferenceProperty annotationIntrospector_ReferenceProperty0 = AnnotationIntrospector.ReferenceProperty.back(">c-S");
      boolean boolean0 = annotationIntrospector_ReferenceProperty0.isBackReference();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotationIntrospector.ReferenceProperty.Type[] annotationIntrospector_ReferenceProperty_TypeArray0 = AnnotationIntrospector.ReferenceProperty.Type.values();
      Class<NamedType> class0 = NamedType.class;
      String[] stringArray0 = new String[20];
      stringArray0[1] = "";
      String[] stringArray1 = annotationIntrospector0.findEnumValues(class0, annotationIntrospector_ReferenceProperty_TypeArray0, stringArray0);
      assertEquals(20, stringArray1.length);
  }
}
