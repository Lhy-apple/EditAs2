/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:44:25 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.databind.jsontype.impl.StdSubtypeResolver;
import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdSubtypeResolver_ESTest extends StdSubtypeResolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      LinkedHashSet<NamedType> linkedHashSet0 = new LinkedHashSet<NamedType>();
      stdSubtypeResolver0._registeredSubtypes = linkedHashSet0;
      Class<ObjectIdGenerators.UUIDGenerator>[] classArray0 = (Class<ObjectIdGenerators.UUIDGenerator>[]) Array.newInstance(Class.class, 0);
      stdSubtypeResolver0.registerSubtypes(classArray0);
      assertEquals(0, classArray0.length);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      Class<ObjectIdGenerators.UUIDGenerator>[] classArray0 = (Class<ObjectIdGenerators.UUIDGenerator>[]) Array.newInstance(Class.class, 0);
      stdSubtypeResolver0.registerSubtypes(classArray0);
      assertEquals(0, classArray0.length);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      NamedType[] namedTypeArray0 = new NamedType[1];
      stdSubtypeResolver0.registerSubtypes(namedTypeArray0);
      assertEquals(1, namedTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      Class<Locale.FilteringMode>[] classArray0 = (Class<Locale.FilteringMode>[]) Array.newInstance(Class.class, 3);
      // Undeclared exception!
      try { 
        stdSubtypeResolver0.registerSubtypes(classArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.NamedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      Class<Object> class0 = Object.class;
      NamedType namedType0 = new NamedType(class0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertEquals(1, hashMap0.size());
      assertFalse(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      Class<Object> class0 = Object.class;
      NamedType namedType0 = new NamedType(class0);
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      NamedType namedType1 = new NamedType(class0, "-]vsb]s31Nx{y");
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType1, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertNotSame(namedType1, namedType0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      HashMap<NamedType, NamedType> hashMap0 = new HashMap<NamedType, NamedType>();
      StdSubtypeResolver stdSubtypeResolver0 = new StdSubtypeResolver();
      Class<Locale.FilteringMode> class0 = Locale.FilteringMode.class;
      NamedType namedType0 = new NamedType(class0, "WRITE_NULL_MAP_VALUES");
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      stdSubtypeResolver0._collectAndResolve((AnnotatedClass) null, namedType0, (MapperConfig<?>) null, annotationIntrospector0, hashMap0);
      assertEquals("WRITE_NULL_MAP_VALUES", namedType0.getName());
  }
}
