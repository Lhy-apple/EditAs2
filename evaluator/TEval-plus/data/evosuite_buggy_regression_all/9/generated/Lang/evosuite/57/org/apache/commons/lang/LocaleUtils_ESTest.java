/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:31:29 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.lang.LocaleUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocaleUtils_ESTest extends LocaleUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocaleUtils localeUtils0 = new LocaleUtils();
      List list0 = LocaleUtils.languagesByCountry("TH");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("rf_OC_b~");
      assertNotNull(locale0);
      assertEquals("rf", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("fr");
      assertEquals("", locale0.getISO3Country());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("rf_OC");
      assertEquals("rf_OC", locale0.toString());
      assertNotNull(locale0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("D!H]j");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: D!H]j
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("}-k}'kQe+#g>R7Sa 68");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: }-k}'kQe+#g>R7Sa 68
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("m4Q108zXU7{4GpIkQ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: m4Q108zXU7{4GpIkQ
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("e^\"@0?S#a(1sc^7+");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: e^\"@0?S#a(1sc^7+
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("zja/J7WL!n39vqPTN{D");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: zja/J7WL!n39vqPTN{D
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("rf_0b~i,{");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: rf_0b~i,{
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("rf_i,b~i,{");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: rf_i,b~i,{
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("rf_C9~,)");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: rf_C9~,)
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("rf_Cb~i,{");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: rf_Cb~i,{
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("rf_MC[b~i,{");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: rf_MC[b~i,{
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      List list0 = LocaleUtils.localeLookupList((Locale) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.PRC;
      Locale locale1 = Locale.JAPANESE;
      List list0 = LocaleUtils.localeLookupList(locale0, locale1);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Set set0 = LocaleUtils.availableLocaleSet();
      Set set1 = LocaleUtils.availableLocaleSet();
      assertSame(set1, set0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Set set0 = LocaleUtils.availableLocaleSet();
      assertNotNull(set0);
      
      Locale locale0 = Locale.ITALY;
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      LocaleUtils.availableLocaleSet();
      Locale locale0 = new Locale("QZ_");
      locale0.getDisplayScript();
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertFalse(boolean0);
      
      List list0 = LocaleUtils.localeLookupList(locale0, locale0);
      assertEquals(1, list0.size());
      
      List list1 = LocaleUtils.languagesByCountry("");
      assertEquals(46, list1.size());
      assertNotNull(list1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LocaleUtils.availableLocaleList();
      LocaleUtils.toLocale((String) null);
      List list0 = LocaleUtils.availableLocaleList();
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      List list1 = LocaleUtils.languagesByCountry((String) null);
      assertFalse(list1.equals((Object)list0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LocaleUtils.countriesByLanguage("}-k}'kQe+#g>R7Sa 68");
      String string0 = "DL51?l7b>I5";
      Locale locale0 = new Locale("", "DL51?l7b>I5", "DL51?l7b>I5");
      LocaleUtils.localeLookupList(locale0);
      LocaleUtils.countriesByLanguage("}-k}'kQe+#g>R7Sa 68");
      List list0 = LocaleUtils.countriesByLanguage("");
      LocaleUtils.countriesByLanguage("}-k}'kQe+#g>R7Sa 68");
      Locale locale1 = Locale.ITALIAN;
      locale1.getDisplayScript();
      locale1.getDisplayName(locale1);
      LinkedList<Locale.LanguageRange> linkedList0 = new LinkedList<Locale.LanguageRange>();
      LinkedList<String> linkedList1 = new LinkedList<String>();
      linkedList0.containsAll(list0);
      linkedList1.iterator();
      linkedList0.poll();
      locale1.getDisplayVariant(locale1);
      locale1.getDisplayScript();
      locale1.getUnicodeLocaleAttributes();
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LocaleUtils.countriesByLanguage((String) null);
      Locale locale0 = Locale.PRC;
      Locale locale1 = Locale.JAPANESE;
      assertFalse(locale1.equals((Object)locale0));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LocaleUtils localeUtils0 = new LocaleUtils();
      Locale locale0 = Locale.JAPANESE;
      LinkedList<Locale.LanguageRange> linkedList0 = new LinkedList<Locale.LanguageRange>();
      List list0 = LocaleUtils.countriesByLanguage("th");
      assertFalse(list0.isEmpty());
  }
}