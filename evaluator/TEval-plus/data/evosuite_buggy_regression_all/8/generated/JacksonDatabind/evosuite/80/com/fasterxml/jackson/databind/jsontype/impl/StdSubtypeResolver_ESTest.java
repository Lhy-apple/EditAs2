/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:17:06 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import java.lang.reflect.Array;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdSubtypeResolver_ESTest extends StdSubtypeResolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      Class<Integer>[] classArray0 = (Class<Integer>[]) Array.newInstance(Class.class, 1);
      Class<Integer> class0 = Integer.class;
      classArray0[0] = class0;
      stdSubtypeResolver0.registerSubtypes(classArray0);
      NamedType[] namedTypeArray0 = new NamedType[1];
      stdSubtypeResolver0.registerSubtypes(namedTypeArray0);
      assertEquals(1, namedTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<String> class0 = String.class;
      NamedType namedType0 = new NamedType(class0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertFalse(hashMap0.isEmpty());
      assertEquals(1, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<String> class0 = String.class;
      NamedType namedType0 = new NamedType(class0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      NamedType namedType1 = new NamedType(class0, "com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver");
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType1, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertTrue(namedType1.hasName());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      Class<NamedType> class0 = NamedType.class;
      NamedType namedType0 = new NamedType(class0, "SORT_PROPERTIES_ALPHABETICALLY");
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      hashMap0.put(namedType0, namedType0);
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertTrue(namedType0.hasName());
  }
}
