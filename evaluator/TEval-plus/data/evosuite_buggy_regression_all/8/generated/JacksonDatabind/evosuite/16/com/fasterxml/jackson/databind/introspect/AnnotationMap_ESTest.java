/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:01:54 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import java.lang.annotation.Annotation;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnnotationMap_ESTest extends AnnotationMap_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = annotationMap0.get(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      Class<Annotation> class0 = Annotation.class;
      Annotation annotation0 = annotationMap0.get(class0);
      assertNull(annotation0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      Iterable<Annotation> iterable0 = annotationMap0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      Iterable<Annotation> iterable0 = annotationMap0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      Class<Annotation> class0 = Annotation.class;
      hashMap0.put(class0, (Annotation) null);
      annotationMap0._annotations = hashMap0;
      Iterable<Annotation> iterable0 = annotationMap0.annotations();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AnnotationMap annotationMap0 = AnnotationMap.merge((AnnotationMap) null, (AnnotationMap) null);
      assertNull(annotationMap0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      AnnotationMap annotationMap1 = AnnotationMap.merge(annotationMap0, annotationMap0);
      assertEquals(0, annotationMap1.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      Class<Annotation> class0 = Annotation.class;
      hashMap0.put(class0, (Annotation) null);
      // Undeclared exception!
      try { 
        AnnotationMap.merge(annotationMap0, annotationMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotationMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      Class<Annotation> class0 = Annotation.class;
      hashMap0.put(class0, (Annotation) null);
      AnnotationMap annotationMap1 = AnnotationMap.merge(annotationMap0, (AnnotationMap) null);
      assertEquals(1, annotationMap1.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      Class<Annotation> class0 = Annotation.class;
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      hashMap0.put(class0, (Annotation) null);
      AnnotationMap annotationMap1 = new AnnotationMap();
      AnnotationMap annotationMap2 = AnnotationMap.merge(annotationMap0, annotationMap1);
      assertEquals(1, annotationMap2.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      Class<Annotation> class0 = Annotation.class;
      hashMap0.put(class0, (Annotation) null);
      AnnotationMap annotationMap1 = new AnnotationMap();
      AnnotationMap annotationMap2 = AnnotationMap.merge(annotationMap1, annotationMap1);
      HashMap<Class<Annotation>, Annotation> hashMap1 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap2._annotations = hashMap1;
      AnnotationMap annotationMap3 = AnnotationMap.merge(annotationMap0, annotationMap2);
      assertEquals(1, annotationMap3.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      int int0 = annotationMap0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      int int0 = annotationMap0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      // Undeclared exception!
      try { 
        annotationMap0.addIfNotPresent((Annotation) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotationMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      // Undeclared exception!
      try { 
        annotationMap0.addIfNotPresent((Annotation) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotationMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      String string0 = annotationMap0.toString();
      assertEquals("{}", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      String string0 = annotationMap0.toString();
      assertEquals("[null]", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      AnnotationMap annotationMap0 = new AnnotationMap();
      HashMap<Class<Annotation>, Annotation> hashMap0 = new HashMap<Class<Annotation>, Annotation>();
      annotationMap0._annotations = hashMap0;
      // Undeclared exception!
      try { 
        annotationMap0.add((Annotation) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.AnnotationMap", e);
      }
  }
}